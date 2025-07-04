import torch
import triton
import triton.language as tl

# -------------------
# Triton Kernels
# -------------------

@triton.jit
def _softmax_forward_kernel(
    output_ptr, input_ptr,
    input_stride_row, output_stride_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for the forward pass of the Softmax function.
    
    Processes one row of the input tensor per program instance.
    
    Args:
        output_ptr: Pointer to the output tensor.
        input_ptr: Pointer to the input tensor.
        input_stride_row: Stride for moving between rows in the input tensor.
        output_stride_row: Stride for moving between rows in the output tensor.
        n_cols: Number of columns in the input tensor.
        BLOCK_SIZE: The smallest power of 2 greater than or equal to n_cols.
                    This is used for efficient memory access.
    """
    # 1. Get the program ID, which corresponds to the row index.
    row_idx = tl.program_id(axis=0)

    # 2. Calculate pointers to the start of the current row for input and output.
    row_start_ptr = input_ptr + row_idx * input_stride_row
    
    # 3. Create column offsets for the current row.
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # 4. Create a mask to prevent out-of-bounds memory access for rows
    #    where n_cols is not a power of 2.
    input_pointers = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    # 5. Load the row of data from HBM (High Bandwidth Memory) to SRAM (Static RAM).
    #    Use the mask to safely load data, padding with -inf where the mask is false.
    row = tl.load(input_pointers, mask=mask, other=-float('inf'))

    # 6. Compute softmax.
    #    - Subtract the maximum value for numerical stability (prevents overflow).
    #    - Exponentiate the values.
    #    - Normalize by dividing by the sum of exponentiated values.
    safe_row = row - tl.max(row, axis=0)
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # 7. Write the result back to HBM.
    output_pointers = output_ptr + row_idx * output_stride_row + col_offsets
    tl.store(output_pointers, softmax_output, mask=mask)


@triton.jit
def _softmax_backward_kernel(
    
    grad_input_ptr, grad_output_ptr, output_ptr,
    grad_stride_row, output_stride_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for the backward pass of the Softmax function.
    
    Computes the Jacobian-vector product: dx = (J_softmax @ dy).
    
    Args:
        grad_input_ptr: Pointer to the gradient of the input (dx).
        grad_output_ptr: Pointer to the gradient of the output (dy).
        output_ptr: Pointer to the output of the forward pass (y).
        grad_stride_row: Stride for moving between rows in the gradient tensors.
        output_stride_row: Stride for moving between rows in the output tensor.
        n_cols: Number of columns in the tensors.
        BLOCK_SIZE: The smallest power of 2 greater than or equal to n_cols.
    """
    # 1. Get the program ID, which corresponds to the row index.
    row_idx = tl.program_id(axis=0)

    # 2. Create column offsets for the current row.
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # 3. Calculate pointers for the current row in the output (y) and grad_output (dy) tensors.
    output_row_ptr = output_ptr + row_idx * output_stride_row + col_offsets
    grad_output_row_ptr = grad_output_ptr + row_idx * grad_stride_row + col_offsets

    # 4. Load the corresponding rows of y and dy from HBM to SRAM.
    y = tl.load(output_row_ptr, mask=mask, other=0.0)
    dy = tl.load(grad_output_row_ptr, mask=mask, other=0.0)

    # 5. Compute the Jacobian-vector product.
    #    The formula is: dx = y * (dy - sum(y * dy)).
    #    - Calculate the dot product of y and dy for the row.
    dot_product = tl.sum(y * dy, axis=0)
    
    #    - Compute the gradient of the input.
    grad_input = y * (dy - dot_product)

    # 6. Write the result (dx) back to HBM.
    grad_input_ptr = grad_input_ptr + row_idx * grad_stride_row + col_offsets
    tl.store(grad_input_ptr, grad_input, mask=mask)


# -------------------
# PyTorch autograd.Function
# -------------------

class Softmax(torch.autograd.Function):
    """
    A custom Softmax activation function implemented using Triton kernels for
    improved performance on CUDA GPUs.
    """
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass for the Softmax function.
        
        Args:
            x (torch.Tensor): A 2D tensor of shape (M, N).
        
        Returns:
            torch.Tensor: The result of softmax(x).
        """
        # Input validation
        if not x.is_cuda:
            raise ValueError("Input tensor must be on a CUDA device.")
        if x.dim() != 2:
            raise ValueError("Input tensor must be 2D.")
        if not x.is_contiguous():
            x = x.contiguous()

        n_rows, n_cols = x.shape
        
        # Determine the block size for the Triton kernel.
        # It must be a power of 2 and >= n_cols.
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        
        # Allocate the output tensor.
        y = torch.empty_like(x)
        
        # Define the grid for launching the Triton programs.
        # We launch one program for each row of the input tensor.
        grid = (n_rows,)
        
        # Launch the forward kernel.
        _softmax_forward_kernel[grid](
            y, x,
            x.stride(0), y.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Save the output tensor for use in the backward pass.
        ctx.save_for_backward(y)
        
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the Softmax function.
        
        Args:
            grad_output (torch.Tensor): The gradient of the loss with respect to the output.
        
        Returns:
            torch.Tensor: The gradient of the loss with respect to the input.
        """
        # Input validation
        if not grad_output.is_cuda:
            raise ValueError("Gradient tensor must be on a CUDA device.")
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
            
        # Retrieve the saved output tensor from the forward pass.
        y, = ctx.saved_tensors
        
        # Allocate tensor for the input gradients.
        grad_input = torch.empty_like(grad_output)
        
        n_rows, n_cols = grad_output.shape
        
        # Determine the block size for the Triton kernel.
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        
        # Define the grid for launching the Triton programs.
        grid = (n_rows,)
        
        # Launch the backward kernel.
        _softmax_backward_kernel[grid](
            grad_input, grad_output, y,
            grad_output.stride(0), y.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return grad_input


# -------------------
# Main Execution Block
# -------------------

if __name__ == "__main__":
    # Expose the custom function for use
    softmax = Softmax.apply

    # --- Test Configuration ---
    torch.manual_seed(0)
    # Use a non-power-of-2 dimension to test masking
    rows, cols = 16, 500  
    device = 'cuda'
    
    # Create input tensors. One for Triton, one for PyTorch's native implementation.
    input_tensor = torch.randn(rows, cols, device=device, requires_grad=True)
    input_clone = input_tensor.clone().detach().requires_grad_(True)
    
    print(f"--- Testing Softmax Implementation on {device} ---")
    print(f"Tensor Shape: ({rows}, {cols})\n")

    # --- Triton Implementation Test ---
    print("1. Running Triton implementation...")
    output_triton = softmax(input_tensor)
    
    # Simulate a downstream gradient and run backward pass
    downstream_grad = torch.randn_like(output_triton)
    output_triton.backward(downstream_grad)
    dx_triton = input_tensor.grad
    print("   Triton forward and backward passes complete.\n")

    # --- PyTorch Native Implementation Test ---
    print("2. Running PyTorch native implementation...")
    output_torch = torch.nn.functional.softmax(input_clone, dim=-1)
    
    # Run backward pass with the same downstream gradient
    output_torch.backward(downstream_grad)
    dx_torch = input_clone.grad
    print("   PyTorch forward and backward passes complete.\n")

    # --- Comparison ---
    print("3. Comparing results...")
    # Use a small tolerance for floating point comparisons
    atol = 1e-5 
    forward_pass_close = torch.allclose(output_triton, output_torch, atol=atol)
    backward_pass_close = torch.allclose(dx_triton, dx_torch, atol=atol)

    print(f"   Forward pass is close: {forward_pass_close}")
    print(f"   Backward pass is close: {backward_pass_close}\n")

    if forward_pass_close and backward_pass_close:
        print("✅ All tests passed!")
    else:
        print("❌ Tests failed. Discrepancy found between Triton and PyTorch implementations.")

