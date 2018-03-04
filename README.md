# MagicaMenger

A Python/numpy implementation of the Menger Sponge/Cube designed to work with MagicaVoxel.

<img src="https://raw.githubusercontent.com/mathewa6/MagicaMenger/master/images/image3.png" align = "center">

## Modules

- [pyvox](https://github.com/gromgull/py-vox-io)
- [numpy](http://www.numpy.org)

## Functionality

- Set ```object.lut = []``` to indicate which voxels in a 3x3x3 cube to delete. (0 ... 26)
- Call ```output(depth, filename)``` to return a numpy array of the traditional Menger sponge as well as write it to filename.vox.
- Use ```inverseOutput(depth, filename)``` to output the model of all deleted cubes in the regular sponge.

## Usage

```python
# Create an object 
menger = MengerMagica()

# Call either output() or inverseOutput()
invop = menger.inverseOutput(3, "output.vox")
```

## Notes

- Calling inverseOutput results in an internal call to ```output()```
- Calculations for a call to ```output()``` for a certain depth occur once. Changing the depth results in recomputation.

## Based on
The method of using a look up table for voxel deletion was borrowed from [Malcolm Kesson](http://www.fundza.com/algorithmic/menger/index.html).
