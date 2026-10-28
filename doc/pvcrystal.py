# state file generated using paraview version 5.13.0
import paraview
import sys
size, time = sys.argv[1], sys.argv[2]
paraview.compatibility.major = 5
paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [2084, 939]
renderView1.InteractionMode = '2D'
renderView1.AxesGrid = 'Grid Axes 3D Actor'
renderView1.OrientationAxesVisibility = 0
renderView1.CenterOfRotation = [6.0, 6.0, 0.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [8.306853554456245, 6.030259125751028, 13.4]
renderView1.CameraFocalPoint = [8.306853554456245, 6.030259125751028, 0.0]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 1.9318537837211862
renderView1.LegendGrid = 'Legend Grid Actor'
renderView1.PolarGrid = 'Polar Grid Actor'
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(2084, 939)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Partitioned Unstructured Grid Reader'
s0003crystal00000pvtu = XMLPartitionedUnstructuredGridReader(registrationName='s0003-crystal00000.pvtu*',
                        FileName=[f's000{size}-crystal{time.zfill(5)}.pvtu'])
s0003crystal00000pvtu.CellArrayStatus = ['levels', 'rank']
s0003crystal00000pvtu.PointArrayStatus = ['solution']
s0003crystal00000pvtu.TimeArray = 'None'

# create a new 'Transform'
transform1 = Transform(registrationName='Transform1', Input=s0003crystal00000pvtu)
transform1.Transform = 'Transform'

# init the 'Transform' selected for 'Transform'
transform1.Transform.Translate = [4.1, 0.0, 0.0]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from s0003crystal00000pvtu
s0003crystal00000pvtuDisplay = Show(s0003crystal00000pvtu, renderView1, 'UnstructuredGridRepresentation')

# get 2D transfer function for 'solution'
solutionTF2D = GetTransferFunction2D('solution')

# get color transfer function/color map for 'solution'
solutionLUT = GetColorTransferFunction('solution')
solutionLUT.TransferFunction2D = solutionTF2D
solutionLUT.RGBPoints = [-0.0005120408604852855, 0.0, 0.0, 0.5625, 0.11069048939828004, 0.0, 0.0, 1.0, 0.3648682018302015, 0.0, 1.0, 1.0, 0.4919568078402189, 0.5, 1.0, 0.5, 0.6190454138502361, 1.0, 1.0, 0.0, 0.873223126282158, 1.0, 0.0, 0.0, 1.0003117322921753, 0.5, 0.0, 0.0]
solutionLUT.ColorSpace = 'RGB'
solutionLUT.ScalarRangeInitialized = 1.0
solutionLUT.VectorMode = 'Component'

# get opacity transfer function/opacity map for 'solution'
solutionPWF = GetOpacityTransferFunction('solution')
solutionPWF.Points = [-0.0005120408604852855, 0.0, 0.5, 0.0, 1.0003117322921753, 1.0, 0.5, 0.0]
solutionPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
s0003crystal00000pvtuDisplay.Representation = 'Surface'
s0003crystal00000pvtuDisplay.ColorArrayName = ['POINTS', 'solution']
s0003crystal00000pvtuDisplay.LookupTable = solutionLUT
s0003crystal00000pvtuDisplay.SelectNormalArray = 'None'
s0003crystal00000pvtuDisplay.SelectTangentArray = 'None'
s0003crystal00000pvtuDisplay.SelectTCoordArray = 'None'
s0003crystal00000pvtuDisplay.TextureTransform = 'Transform2'
s0003crystal00000pvtuDisplay.OSPRayScaleArray = 'solution'
s0003crystal00000pvtuDisplay.OSPRayScaleFunction = 'Piecewise Function'
s0003crystal00000pvtuDisplay.Assembly = ''
s0003crystal00000pvtuDisplay.SelectedBlockSelectors = ['']
s0003crystal00000pvtuDisplay.SelectOrientationVectors = 'None'
s0003crystal00000pvtuDisplay.ScaleFactor = 0.4
s0003crystal00000pvtuDisplay.SelectScaleArray = 'levels'
s0003crystal00000pvtuDisplay.GlyphType = 'Arrow'
s0003crystal00000pvtuDisplay.GlyphTableIndexArray = 'solution'
s0003crystal00000pvtuDisplay.GaussianRadius = 0.02
s0003crystal00000pvtuDisplay.SetScaleArray = ['POINTS', 'solution']
s0003crystal00000pvtuDisplay.ScaleTransferFunction = 'Piecewise Function'
s0003crystal00000pvtuDisplay.OpacityArray = ['POINTS', 'solution']
s0003crystal00000pvtuDisplay.OpacityTransferFunction = 'Piecewise Function'
s0003crystal00000pvtuDisplay.DataAxesGrid = 'Grid Axes Representation'
s0003crystal00000pvtuDisplay.PolarAxes = 'Polar Axes Representation'
s0003crystal00000pvtuDisplay.ScalarOpacityFunction = solutionPWF
s0003crystal00000pvtuDisplay.ScalarOpacityUnitDistance = 0.5237428905769653
s0003crystal00000pvtuDisplay.OpacityArrayName = ['POINTS', 'solution']
s0003crystal00000pvtuDisplay.SelectInputVectors = [None, '']
s0003crystal00000pvtuDisplay.WriteLog = ''

# init the 'Piecewise Function' selected for 'OSPRayScaleFunction'
s0003crystal00000pvtuDisplay.OSPRayScaleFunction.Points = [-1.0386152267456055, 0.0, 0.5, 0.0, 0.25505340099334717, 1.0, 0.5, 0.0]

# show data from transform1
transform1Display = Show(transform1, renderView1, 'UnstructuredGridRepresentation')

# get separate 2D transfer function for 'rank'
separate_transform1Display_rankTF2D = GetTransferFunction2D('rank', transform1Display, separate=True)

# get separate color transfer function/color map for 'rank'
separate_transform1Display_rankLUT = GetColorTransferFunction('rank', transform1Display, separate=True)
separate_transform1Display_rankLUT.InterpretValuesAsCategories = 1
separate_transform1Display_rankLUT.AnnotationsInitialized = 1
separate_transform1Display_rankLUT.TransferFunction2D = separate_transform1Display_rankTF2D
separate_transform1Display_rankLUT.RGBPoints = [0.0, 0.0, 0.0, 0.5625, 0.222222, 0.0, 0.0, 1.0, 0.7301590000000001, 0.0, 1.0, 1.0, 0.984127, 0.5, 1.0, 0.5, 1.238095, 1.0, 1.0, 0.0, 1.746032, 1.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0]
separate_transform1Display_rankLUT.ColorSpace = 'RGB'
separate_transform1Display_rankLUT.ScalarRangeInitialized = 1.0
separate_transform1Display_rankLUT.Annotations = ['0', '0', '1', '1', '2', '2']
separate_transform1Display_rankLUT.IndexedColors = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
separate_transform1Display_rankLUT.IndexedOpacities = [1.0, 1.0, 1.0]
separate_transform1Display_rankLUT.EnableOpacityMapping = 1

# get separate opacity transfer function/opacity map for 'rank'
separate_transform1Display_rankPWF = GetOpacityTransferFunction('rank', transform1Display, separate=True)
separate_transform1Display_rankPWF.Points = [0.0, 0.0, 0.5, 0.0, 2.0, 1.0, 0.5, 0.0]
separate_transform1Display_rankPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
transform1Display.Representation = 'Surface With Edges'
transform1Display.ColorArrayName = ['CELLS', 'rank']
transform1Display.LookupTable = separate_transform1Display_rankLUT
transform1Display.SelectNormalArray = 'None'
transform1Display.SelectTangentArray = 'None'
transform1Display.SelectTCoordArray = 'None'
transform1Display.TextureTransform = 'Transform2'
transform1Display.EdgeColor = [0.796078431372549, 0.796078431372549, 0.796078431372549]
transform1Display.OSPRayScaleArray = 'solution'
transform1Display.OSPRayScaleFunction = 'Piecewise Function'
transform1Display.Assembly = ''
transform1Display.SelectedBlockSelectors = ['']
transform1Display.SelectOrientationVectors = 'None'
transform1Display.ScaleFactor = 0.4
transform1Display.SelectScaleArray = 'None'
transform1Display.GlyphType = 'Arrow'
transform1Display.GlyphTableIndexArray = 'solution'
transform1Display.GaussianRadius = 0.02
transform1Display.SetScaleArray = ['POINTS', 'solution']
transform1Display.ScaleTransferFunction = 'Piecewise Function'
transform1Display.OpacityArray = ['POINTS', 'solution']
transform1Display.OpacityTransferFunction = 'Piecewise Function'
transform1Display.DataAxesGrid = 'Grid Axes Representation'
transform1Display.PolarAxes = 'Polar Axes Representation'
transform1Display.ScalarOpacityFunction = separate_transform1Display_rankPWF
transform1Display.ScalarOpacityUnitDistance = 0.5237428905769653
transform1Display.OpacityArrayName = ['POINTS', 'solution']
transform1Display.SelectInputVectors = [None, '']
transform1Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'OSPRayScaleFunction'
transform1Display.OSPRayScaleFunction.Points = [-1.0386152267456055, 0.0, 0.5, 0.0, 0.25505340099334717, 1.0, 0.5, 0.0]

# set separate color map
transform1Display.UseSeparateColorMap = True

# setup the color legend parameters for each legend in this view

# get color legend/bar for separate_transform1Display_rankLUT in view renderView1
separate_transform1Display_rankLUTColorBar = GetScalarBar(separate_transform1Display_rankLUT, renderView1)
separate_transform1Display_rankLUTColorBar.WindowLocation = 'Any Location'
separate_transform1Display_rankLUTColorBar.Position = [0.9618426103646832, 0.3375931842385516]
separate_transform1Display_rankLUTColorBar.Title = 'rank'
separate_transform1Display_rankLUTColorBar.ComponentTitle = ''
separate_transform1Display_rankLUTColorBar.ScalarBarLength = 0.3299999999999999

# set color bar visibility
separate_transform1Display_rankLUTColorBar.Visibility = 1

# show color legend
transform1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get time animation track
timeAnimationCue1 = GetTimeTrack()

# initialize the animation scene

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# initialize the timekeeper

# initialize the animation track

# get animation scene
animationScene1 = GetAnimationScene()

# initialize the animation scene
animationScene1.ViewModules = renderView1
animationScene1.Cues = timeAnimationCue1
animationScene1.AnimationTime = 0.0
animationScene1.EndTime = 5.0
animationScene1.PlayMode = 'Snap To TimeSteps'

# ----------------------------------------------------------------
# restore active source
SetActiveSource(s0003crystal00000pvtu)
# ----------------------------------------------------------------

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [900, 400]
Render()
ResetCamera(view)
renderView1.CameraParallelScale = 2.1
SaveScreenshot(f"crystal{size}_{time}.png", view)


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, useful when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple split view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://www.paraview.org/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------
