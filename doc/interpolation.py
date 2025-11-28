from paraview.simple import *

data = XMLUnstructuredGridReader(FileName=['interpolation_subsampled.vtu'])

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1036, 768]
if GetParaViewVersion() < 5.5:
    view.UseOffscreenRendering = True

view.CenterAxesVisibility = 0
view.OrientationAxesVisibility = 0

# this does not work with newer ParaView versions
try:
    view.UseGradientBackground = False
except Exception:
    pass

view.Background = [1.0, 1.0, 1.0]

camera = view.GetActiveCamera()
camera.SetPosition(1.5, -1.5, 3)
camera.SetFocalPoint(-0.25, 0.25, 0.0)
camera.SetViewUp(0.0, 0.0, 1.0)
camera.Zoom(0.8)

warped = WarpByScalar(Input=data)

for current in ['discrete', 'exact', 'error']:
    rg = tuple(round(v, 2) for v in data.PointData.GetArray(current).GetRange())

    warped.Scalars = ['POINTS', current]
    warped.ScaleFactor = 0.5
    display = Show(warped, view)

    display.ColorArrayName = ('POINTS', current)
    display.LookupTable = GetLookupTableForArray(current, 1, Discretize=0, RGBPoints=[rg[0], 0.0, 0.0, 1.0, rg[1], 1.0, 0.0, 0.0], ColorSpace='HSV')
    display.SetRepresentationType('Surface with Edges')
    display.EdgeColor = [1./3., 1./3., 1./3.]

    colorbar = GetScalarBar(display.LookupTable, view)
    colorbar.Position = [0.9, 0.2]
    # colorbar.Position2 = [0.1, 0.6]
    colorbar.LabelFontSize = 12
    colorbar.LabelColor = [0.0, 0.0, 0.0]
    colorbar.AutomaticLabelFormat = 0
    colorbar.LabelFormat = "%0.2f"
    colorbar.RangeLabelFormat = "%0.2f"
    colorbar.Orientation = 'Horizontal'
    display.SetScalarBarVisibility(view, True)

    SaveScreenshot('figures/interpolation_%s.png' % current, magnification=1, quality=100, view=view)

    Hide(warped, view)
