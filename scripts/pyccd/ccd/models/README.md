# Cython Lasso Implementation for PyCCD

## Installation

### Prerequisites

Before compiling the Cython extension, ensure you have the following installed:
- Python 3.6+
- NumPy
- Cython
- A C compiler (GCC, Clang, MSVC, etc.)
- scikit-learn (for comparison and fallback)

Install the required Python packages:

```bash
pip install numpy cython scikit-learn
```

### Compiling the Cython Extension

Run this command to compile the Cython extension:

```bash
python setup.py build_ext --inplace
```

This will compile the Cython code and create a platform-specific binary extension that can be imported in Python.

## Usage

### Integration with PyCCD

The `lasso.py` module will automatically use the Cython implementation if available:

```python
from ccd.models.lasso import fitted_model, coefficient_matrix

# Use the same interface as before
model = fitted_model(dates, spectra_obs, max_iter, avg_days_yr, num_coefficients, alpha)
```

## Cross-Platform Considerations

This Cython extension needs to be compiled on each platform separately. The source code (.pyx) is platform-independent, but the compiled binary (.so, .pyd) is specific to the operating system and Python version.

When sharing this code with team members on different platforms:
1. Do not commit compiled binaries (.so, .pyd, .o, .c) to version control
2. Each team member should run the compilation step on their own machine
3. Use the provided setup.py for consistent compilation across platforms

## Testing

There is a test script in the pyccd/ directory that can be ran to ensure the Cython Lasso is working:

```bash
python test_cython_lasso.py
```

This will output timing information and coefficient comparisons between the Cython implementation and scikit-learn for a random data set.

## Troubleshooting

If you encounter issues with the Cython implementation:

1. **ImportError for cython_lasso**: Ensure you've compiled the extension with `python setup.py build_ext --inplace`

2. **Compilation errors**: Make sure you have a C compiler installed and properly configured

3. **Performance issues**: Try adjusting the `alpha` parameter for your specific dataset

4. **Different results from scikit-learn**: Small differences in coefficients are normal due to implementation differences in the coordinate descent algorithm