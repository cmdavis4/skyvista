# Pyvista setup tips

Setting up pyvista can be a bit onerous depending on your specific configuration—whether you're using a remote machine, whether you're using jupyter from within VSCode, etc. The [PyVista installation docs](https://docs.pyvista.org/getting-started/installation), and the [section on remote machines specifically](https://docs.pyvista.org/getting-started/installation#running-on-remote-servers) if that's your use case, are the place to start. With that said, I found that I fell into several gaps following those docs directly, and so I've compiled further setup tips here.

### Testing whether PyVista works
As you work through everything below this point, the simplest way to test if PyVista/skyvista are working is to produce a simple render. Here is some example code that provides a simple test; if you see a bunny and can move/rotate it, you're in business:

```python
from pyvista import examples
dataset = examples.download_bunny()
dataset.plot(cpos='xy')
```

## **Setup required pyvista system libraries**

First install any required system packages from [the documentation](https://docs.pyvista.org/version/stable/getting-started/installation.html#running-on-remote-servers); if running on a remote server, this should be `python-qt4, libgl1-mesa-glx, and xvfb` . You need sudo privileges to install these, and so you may need to ask an administrator to install them. The last time I did this there was some issue installing `python-qt4` that made it seem like it was outdated for the system, but it works fine without it, so I think don’t worry if you run into problems with that one.

Once installed, run the following two commands (or ideally add them to your bash profile):

`export DISPLAY=:99.0`

`export PYVISTA_OFF_SCREEN=true`

Assuming you are running this for the first time, start the X server:

`Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &`

**You will need to run this command every time the server is restarted**.

If you encounter issues, you may need to delete the lock files, which are in `/tmp` and may be called `.X99-lock` or might be in a subdirectory.

## Using jupyter
Most of `skyvista` and its documentation assume you will be running pyvista within jupyter. Instructions will vary based on whether you do this in a web browser vs using the built-in VSCode jupyter client. In general, I think the latter should be preferred (and that is my current setup), and so my notes on how to use it from a browser may be outdated.

### From VSCode

If you try to use PyVista in VSCode jupyter "naively", i.e. assuming it should work without several opaque configuration adjustments, you would be just as foolish as I was. My initial experience of trying to plot PyVista scenes this way was just getting a blank render window. This is especially confounding as there are ultimately several fixes required.

The first fix has to do with port forwarding in some way; I have not dug into the details, but there is discussion of the issue and [the solution](https://github.com/pyvista/pyvista/issues/5296#issuecomment-1971079419) in [an open PyVista bug report from 2023](https://github.com/pyvista/pyvista/issues/5296). All you need to do is set the following VSCode settings to the following values:
- `remote.autoForwardPortsSource`: `process`
- `remote.localPortHost`: `localhost`

The second required fix is after attempting to render a PyVista scene, when you will presumably still get a blank render window. Like the first fix, [the solution](https://github.com/pyvista/pyvista/issues/5296#issuecomment-2374315543) also comes from a comment in an [open PyVista bug report](https://github.com/pyvista/pyvista/issues/5296). The video just linked does a better job showing what to do than I could describe; you just need to toggle the output type of the cell back and forth once. For me, this fix generally persists until I reload the VSCode window, so it's not overly annoying.

### From a browser
When running jupyter within a browser, PyVista rquires the use of the `jupyter_server_proxy` package and some boilerplate. Install the package; you will then need to run the following two lines at the top of every notebook, after importing pyvista/skyvista:

```python
pv.global_theme.trame.server_proxy_enabled = True
pv.global_theme.trame.server_proxy_prefix = "/proxy/"
pv.set_jupyter_backend('trame')
```

and then it should work as normal.
