from xml.dom import minidom
from mujoco_py.utils import remove_empty_lines
from threading import Lock

_MjSim_render_lock = Lock()


cdef class MjSim(object):
    """MjSim represents a running simulation including its state.

    Similar to Gym's ``MujocoEnv``, it internally wraps a :class:`.PyMjModel`
    and a :class:`.PyMjData`.

    Parameters
    ----------
    model : :class:`.PyMjModel`
        The model to simulate.
    data : :class:`.PyMjData`
        Optional container for the simulation state. Will be created if ``None``.
    nsubsteps : int
        Optional number of MuJoCo steps to run for every call to :meth:`.step`.
        Buffers will be swapped only once per step.
    udd_callback : fn(:class:`.MjSim`) -> dict
        Optional callback for user-defined dynamics. At every call to
        :meth:`.step`, it receives an MjSim object ``sim`` containing the
        current user-defined dynamics state in ``sim.udd_state``, and returns the
        next ``udd_state`` after applying the user-defined dynamics. This is
        useful e.g. for reward functions that operate over functions of historical
        state.
    """
    # MjRenderContext for rendering camera views.
    cdef readonly list render_contexts
    cdef readonly object _render_context_window
    cdef readonly object _render_context_offscreen

    # MuJoCo model
    cdef readonly PyMjModel model
    # MuJoCo data
    """
    DATAZ
    """
    cdef readonly PyMjData data
    # Number of substeps when calling .step
    cdef readonly int nsubsteps
    # User defined state.
    cdef readonly dict udd_state
    # User defined dynamics callback
    cdef readonly object _udd_callback
    # Allows to store extra information in MjSim.
    cdef readonly dict extras

    ### MjRenderContext
    cdef mjvScene _offscreen_scn
    cdef mjvCamera _offscreen_cam
    cdef mjvOption _offscreen_vopt
    cdef mjvPerturb _offscreen_pert
    cdef mjrContext _offscreen_con

    # Public wrappers
    cdef          object opengl_context_offscreen

    #####################################
    ### MjRenderContextWindow
    cdef mjModel *_model_ptr
    cdef mjData *_data_ptr

    cdef mjvScene _scn
    cdef mjvCamera _cam
    cdef mjvOption _vopt
    cdef mjvPerturb _pert
    cdef mjrContext _con

    # Public wrappers
    cdef readonly PyMjvScene scn
    cdef readonly PyMjvCamera cam
    cdef readonly PyMjvOption vopt
    cdef readonly PyMjvPerturb pert
    cdef readonly PyMjrContext con

    cdef readonly object opengl_context
    cdef readonly int _visible
    cdef readonly list _markers
    cdef readonly dict _overlay

    cdef readonly bint offscreen
    cdef public object sim
    #####################################

    # cdef public object sim

    def __cinit__(self, PyMjModel model, PyMjData data=None, int nsubsteps=1,
                  udd_callback=None):
        self.nsubsteps = nsubsteps
        self.model = model
        if data is None:
            with wrap_mujoco_warning():
                _data = mj_makeData(self.model.ptr)
            if _data == NULL:
                raise Exception('mj_makeData failed!')
            self.data = WrapMjData(_data, self.model)
        else:
            self.data = data

        self.render_contexts = []
        self._render_context_offscreen = None
        self._render_context_window = None
        self.udd_state = None
        self.udd_callback = udd_callback
        self.extras = {}

        ### MjRenderContext
        maxgeom = 1000
        mjv_makeScene(&self._offscreen_scn, maxgeom)
        mjv_defaultCamera(&self._offscreen_cam)
        mjv_defaultOption(&self._offscreen_vopt)
        mjr_defaultContext(&self._offscreen_con)

        ### MjRenderContext
        self.forward()

        self._offscreen_pert.active = 0
        self._offscreen_pert.select = 0


    def reset(self):
        """
        Resets the simulation data and clears buffers.
        """
        with wrap_mujoco_warning():
            mj_resetData(self.model.ptr, self.data.ptr)

        self.udd_state = None
        self.step_udd()

    def forward(self):
        """
        Computes the forward kinematics. Calls ``mj_forward`` internally.
        """
        with wrap_mujoco_warning():
            mj_forward(self.model.ptr, self.data.ptr)

    def step(self):
        """
        Advances the simulation by calling ``mj_step``.

        If ``qpos`` or ``qvel`` have been modified directly, the user is required to call
        :meth:`.forward` before :meth:`.step` if their ``udd_callback`` requires access to MuJoCo state
        set during the forward dynamics.
        """
        self.step_udd()

        with wrap_mujoco_warning():
            for _ in range(self.nsubsteps):
                mj_step(self.model.ptr, self.data.ptr)

    def render2(self, width=None, height=None, *, camera_name=None, depth=False,
               mode='offscreen', device_id=-1):
        """
        Renders view from a camera and returns image as an `numpy.ndarray`.

        Args:
        - width (int): desired image width.
        - height (int): desired image height.
        - camera_name (str): name of camera in model. If None, the free
            camera will be used.
        - depth (bool): if True, also return depth buffer
        - device (int): device to use for rendering (only for GPU-backed
            rendering).

        Returns:
        - rgb (uint8 array): image buffer from camera
        - depth (float array): depth buffer from camera (only returned
            if depth=True)
        """

        ### render
        cdef mjrRect rect
        rect.left = 0
        rect.bottom = 0
        rect.width = width
        rect.height = height

        ### read_pixels
        rgb_arr = np.zeros(3 * rect.width * rect.height, dtype=np.uint8)
        depth_arr = np.zeros(rect.width * rect.height, dtype=np.float32)
        cdef unsigned char[::view.contiguous] rgb_view = rgb_arr
        cdef float[::view.contiguous] depth_view = depth_arr

        if camera_name is None:
            camera_id = None
        else:
            camera_id = self.model.camera_name2id(camera_name)

        if mode == 'offscreen':
            with _MjSim_render_lock:
                if self._render_context_offscreen is None:
                    maxgeom = 1000
                    mjv_makeScene(&self._offscreen_scn, maxgeom)
                    mjv_defaultCamera(&self._offscreen_cam)
                    mjv_defaultOption(&self._offscreen_vopt)
                    mjr_defaultContext(&self._offscreen_con)

                    device_id = 0 # int(os.getenv('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
                    self.opengl_context_offscreen = OffscreenOpenGLContext(device_id) # TODO

                    mj_forward(self.model.ptr, self.data.ptr)
                    self._render_context_offscreen = self
                    self._offscreen_pert.active = 0
                    self._offscreen_pert.select = 0

                    self._markers = []

                    self._offscreen_cam.type = const.CAMERA_FREE
                    self._offscreen_cam.fixedcamid = -1
                    for i in range(3):
                        self._offscreen_cam.lookat[i] = self.model.stat.center[i]
                    self._offscreen_cam.distance = self.model.stat.extent

                    mjr_makeContext(self.model.ptr, &self._offscreen_con, mjFONTSCALE_150)
                    mjr_setBuffer(mjFB_OFFSCREEN, &self._offscreen_con);
                    if self._offscreen_con.currentBuffer != mjFB_OFFSCREEN:
                        raise RuntimeError('Offscreen rendering not supported')

                self._offscreen_cam.type = const.CAMERA_FREE

                self.opengl_context_offscreen.set_buffer_size(width, height)  # TODO

                mjv_updateScene(self.model.ptr, self.data.ptr, &self._offscreen_vopt,
                                &self._offscreen_pert, &self._offscreen_cam, mjCAT_ALL, &self._offscreen_scn)

                mjr_render(rect, &self._offscreen_scn, &self._offscreen_con)

                mjr_readPixels(&rgb_view[0], &depth_view[0], rect, &self._offscreen_con)


        elif mode == 'window':
            if self._render_context_window is None:
                self._render_context_window = self
                from mujoco_py.builder import cymj

                # render_context = cymj.MjRenderContextWindow(self)
                maxgeom = 1000
                mjv_makeScene(&self._scn, maxgeom)
                mjv_defaultCamera(&self._cam)
                mjv_defaultOption(&self._vopt)
                mjr_defaultContext(&self._con)

                # self.sim = sim
                self._setup_opengl_context(offscreen=False, device_id=device_id)
                self.offscreen = False

                # Ensure the model data has been updated so that there
                # is something to render
                mj_forward(self.model.ptr, self.data.ptr)

                self.add_render_context(self)

                self._model_ptr = self.model.ptr
                self._data_ptr = self.data.ptr
                self.scn = WrapMjvScene(&self._scn)
                self.cam = WrapMjvCamera(&self._cam)
                self.vopt = WrapMjvOption(&self._vopt)
                self.con = WrapMjrContext(&self._con)
                self._pert.active = 0
                self._pert.select = 0
                self.pert = WrapMjvPerturb(&self._pert)

                self._markers = []
                self._overlay = {}

                self._init_camera(self)
                self._set_mujoco_buffers()

            else:
                render_context = self._render_context_window

            # render_context.render()
            if width > self._con.offWidth or height > self._con.offHeight:
                new_width = max(width, self._model_ptr.vis.global_.offwidth)
                new_height = max(height, self._model_ptr.vis.global_.offheight)
                self.update_offscreen_size(new_width, new_height)

            if camera_id is not None:
                if camera_id == -1:
                    self.cam.type = const.CAMERA_FREE
                else:
                    self.cam.type = const.CAMERA_FIXED
                self.cam.fixedcamid = camera_id

            self.opengl_context.set_buffer_size(width, height)

            mjv_updateScene(self._model_ptr, self._data_ptr, &self._vopt,
                            &self._pert, &self._cam, mjCAT_ALL, &self._scn)

            for marker_params in self._markers:
                self._add_marker_to_scene(marker_params)

            mjr_render(rect, &self._scn, &self._con)
            for gridpos, (text1, text2) in self._overlay.items():
                mjr_overlay(const.FONTSCALE_150, gridpos, rect, text1.encode(), text2.encode(), &self._con)

            else:
                raise ValueError("Mode must be either 'window' or 'offscreen'.")

    def render(self, width=None, height=None, *, camera_name=None, depth=False,
               mode='offscreen', device_id=-1):
        """
        Renders view from a camera and returns image as an `numpy.ndarray`.

        Args:
        - width (int): desired image width.
        - height (int): desired image height.
        - camera_name (str): name of camera in model. If None, the free
            camera will be used.
        - depth (bool): if True, also return depth buffer
        - device (int): device to use for rendering (only for GPU-backed
            rendering).

        Returns:
        - rgb (uint8 array): image buffer from camera
        - depth (float array): depth buffer from camera (only returned
            if depth=True)
        """
        if camera_name is None:
            camera_id = None
        else:
            camera_id = self.model.camera_name2id(camera_name)

        if mode == 'offscreen':
            with _MjSim_render_lock:
                if self._render_context_offscreen is None:
                    # render_context = MjRenderContextOffscreen(
                        # self, device_id=device_id)
                    render_context = MjRenderContextOffscreen(
                        self, device_id=device_id)
                else:
                    render_context = self._render_context_offscreen

                render_context.render(
                    width=width, height=height, camera_id=camera_id)

                return render_context.read_pixels(
                    width, height, depth=depth)

        elif mode == 'window':
            if self._render_context_window is None:
                from mujoco_py.mjviewer import MjViewer
                render_context = MjViewer(self)
            else:
                render_context = self._render_context_window

            render_context.render()

        else:
            raise ValueError("Mode must be either 'window' or 'offscreen'.")

    def add_render_context(self, render_context):
        self.render_contexts.append(render_context)
        if render_context.offscreen and self._render_context_offscreen is None:
            self._render_context_offscreen = render_context
        elif not render_context.offscreen and self._render_context_window is None:
            self._render_context_window = render_context

    @property
    def udd_callback(self):
        return self._udd_callback

    @udd_callback.setter
    def udd_callback(self, value):
        self._udd_callback = value
        self.udd_state = None
        self.step_udd()

    def step_udd(self):
        if self._udd_callback is None:
            self.udd_state = {}
        else:
            schema_example = self.udd_state
            self.udd_state = self._udd_callback(self)
            # Check to make sure the udd_state has consistent keys and dimension across steps
            if schema_example is not None:
                keys = set(schema_example.keys()) | set(self.udd_state.keys())
                for key in keys:
                    assert key in schema_example, "Keys cannot be added to udd_state between steps."
                    assert key in self.udd_state, "Keys cannot be dropped from udd_state between steps."
                    if isinstance(schema_example[key], Number):
                        assert isinstance(self.udd_state[key], Number), \
                            "Every value in udd_state must be either a number or a numpy array"
                    else:
                        assert isinstance(self.udd_state[key], np.ndarray), \
                            "Every value in udd_state must be either a number or a numpy array"
                        assert self.udd_state[key].shape == schema_example[key].shape, \
                            "Numpy array values in udd_state must keep the same dimension across steps."

    def get_state(self):
        """ Returns a copy of the simulator state. """
        qpos = np.copy(self.data.qpos)
        qvel = np.copy(self.data.qvel)
        if self.model.na == 0:
            act = None
        else:
            act = np.copy(self.data.act)
        udd_state = copy.deepcopy(self.udd_state)

        return MjSimState(self.data.time, qpos, qvel, act, udd_state)

    def set_state(self, value):
        """
        Sets the state from an MjSimState.
        If the MjSimState was previously unflattened from a numpy array, consider
        set_state_from_flattened, as the defensive copy is a substantial overhead
        in an inner loop.

        Args:
        - value (MjSimState): the desired state.
        - call_forward: optionally call sim.forward(). Called by default if
            the udd_callback is set.
        """
        self.data.time = value.time
        self.data.qpos[:] = np.copy(value.qpos)
        self.data.qvel[:] = np.copy(value.qvel)
        if self.model.na != 0:
            self.data.act[:] = np.copy(value.act)
        self.udd_state = copy.deepcopy(value.udd_state)

    def set_state_from_flattened(self, value):
        """ This helper method sets the state from an array without requiring a defensive copy."""
        state = MjSimState.from_flattened(value, self)

        self.data.time = state.time
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel
        if self.model.na != 0:
            self.data.act[:] = state.act
        self.udd_state = state.udd_state

    def save(self, file, format='xml', keep_inertials=False):
        """
        Saves the simulator model and state to a file as either
        a MuJoCo XML or MJB file. The current state is saved as
        a keyframe in the model file. This is useful for debugging
        using MuJoCo's `simulate` utility.

        Note that this doesn't save the UDD-state which is
        part of MjSimState, since that's not supported natively
        by MuJoCo. If you want to save the model together with
        the UDD-state, you should use the `get_xml` or `get_mjb`
        methods on `MjModel` together with `MjSim.get_state` and
        save them with e.g. pickle.

        Args:
        - file (IO stream): stream to write model to.
        - format: format to use (either 'xml' or 'mjb')
        - keep_inertials (bool): if False, removes all <inertial>
          properties derived automatically for geoms by MuJoco. Note
          that this removes ones that were provided by the user
          as well.
        """
        xml_str = self.model.get_xml()
        dom = minidom.parseString(xml_str)

        mujoco_node = dom.childNodes[0]
        assert mujoco_node.tagName == 'mujoco'

        keyframe_el = dom.createElement('keyframe')
        key_el = dom.createElement('key')
        keyframe_el.appendChild(key_el)
        mujoco_node.appendChild(keyframe_el)

        def str_array(arr):
            return " ".join(map(str, arr))

        key_el.setAttribute('time', str(self.data.time))
        key_el.setAttribute('qpos', str_array(self.data.qpos))
        key_el.setAttribute('qvel', str_array(self.data.qvel))
        if self.data.act is not None:
            key_el.setAttribute('act', str_array(self.data.act))

        if not keep_inertials:
            for element in dom.getElementsByTagName('inertial'):
                element.parentNode.removeChild(element)

        result_xml = remove_empty_lines(dom.toprettyxml(indent=" " * 4))

        if format == 'xml':
            file.write(result_xml)
        elif format == 'mjb':
            new_model = load_model_from_xml(result_xml)
            file.write(new_model.get_mjb())
        else:
            raise ValueError("Unsupported format. Valid ones are 'xml' and 'mjb'")
