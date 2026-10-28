# state file generated using paraview version 5.0.1

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1000,750] # [1514, 662]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
renderView1.CenterOfRotation = [0.0, 0.0, 0.5]
renderView1.StereoType = 0
renderView1.CameraPosition = [-0.7888710491553842, 1.5178694563930326, -1.3228672178327452]
renderView1.CameraFocalPoint = [0.23816691466742906, -0.5240957593629393, 1.120843792335506]
renderView1.CameraViewUp = [-0.4421987040497493, -0.7709512765846979, -0.45836059524041456]
renderView1.CameraParallelScale = 0.8660254037844386
renderView1.Background = [1.0, 1.0, 1.0]
renderView1.CenterAxesVisibility = 0
renderView1.OrientationAxesVisibility = 0

# init the 'GridAxes3DActor' selected for 'AxesGrid'
renderView1.AxesGrid.XTitleColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.YTitleColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.XLabelColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.YLabelColor = [0.0, 0.0, 0.0]
renderView1.AxesGrid.ZLabelColor = [0.0, 0.0, 0.0]

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Unstructured Grid Reader'
a3dexamplevtu = XMLUnstructuredGridReader(FileName=['3dexample.vtu'])
a3dexamplevtu.PointArrayStatus = ['solution']

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get color transfer function/color map for 'solution'
solutionLUT = GetColorTransferFunction('solution')
solutionLUT.RGBPoints = [0.8750640153884888, 0.231373, 0.298039, 0.752941, 1.2335020303726196, 0.865003, 0.865003, 0.865003, 1.5919400453567505, 0.705882, 0.0156863, 0.14902]
solutionLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'solution'
solutionPWF = GetOpacityTransferFunction('solution')
solutionPWF.Points = [0.8750640153884888, 0.0, 0.5, 0.0, 1.5919400453567505, 1.0, 0.5, 0.0]
solutionPWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from a3dexamplevtu
a3dexamplevtuDisplay = Show(a3dexamplevtu, renderView1)
# trace defaults for the display properties.
a3dexamplevtuDisplay.ColorArrayName = ['POINTS', 'solution']
a3dexamplevtuDisplay.LookupTable = solutionLUT
a3dexamplevtuDisplay.EdgeColor = [1.0, 1.0, 1.0]
a3dexamplevtuDisplay.GlyphType = 'Arrow'
a3dexamplevtuDisplay.ScalarOpacityUnitDistance = 0.08521562683627307

SaveScreenshot('figures/3dexample.png', magnification=1, quality=100, view=renderView1)
