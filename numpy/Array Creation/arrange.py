
"""
numpy.arange
numpy.arange([start, ]stop, [step, ]dtype=None)
Return evenly spaced values within a given interval.

Values are generated within the half-open interval [start, stop) (in other words,
the interval including start but excluding stop).
For integer arguments the function is equivalent to the Python built-in range function,
but returns an ndarray rather than a list.

When using a non-integer step, such as 0.1, the results will often not be consistent.
It is better to use linspace for these cases.

Parameters:
start : number, optional

Start of interval. The interval includes this value. The default start value is 0.

stop : number

End of interval. The interval does not include this value, except in some cases where step is not an integer and floating point
round-off affects the length of out.

step : number, optional

Spacing between values. For any output out, this is the distance between two adjacent values, out[i+1] - out[i]. The default step
size is 1. If step is specified, start must also be given.

dtype : dtype

The type of the output array. If dtype is not given, infer the data type from the other input arguments.

Returns:
arange : ndarray

Array of evenly spaced values.

For floating point arguments, the length of the result is ceil((stop - start)/step). Because of floating point overflow,
this rule may result in the last element of out being greater than stop.


"""
import numpy as np

array3 = []
array1 = np.arange(0,3,5)
array2 = np.arange(0,5,2,array3)
print(array1)
print(array2)
print(array3)