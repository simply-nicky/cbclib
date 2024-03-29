Reading Kamzik log files
========================

:class:`LogProtocol <cbclib.LogProtocol>`
-----------------------------------------

Log protocol is a helper class to retrieve the data from the log files generated by the Kamzik server, which contain the read-outs
from the motors and other instruments during the experiment. The data extracted from the log files is used to generate sample
positions and sample orientations during the tilt series by :class:`cbclib.LogContainer`. The protocol consists of the log keys
of the attributes that are required to extract from the header part of a log file and their corresponding data types:

* **datatypes** : Data type of the attributes (`float`, `int`, `str`, or `bool`).
* **log_keys** : Log key to find the attribute in the log file.
* **part_keys** : The name of the part where the attribute is stored in the log file.

.. note::

    You can save protocol to an INI file with :func:`to_ini <cbclib.LogProtocol.to_ini>`
    and import protocol from INI file with :func:`import_ini <cbclib.LogProtocol.import_ini>`.

The default protocol can be accessed with :func:`import_default <cbclib.LogProtocol.import_default>`, which is given by:

.. code-block:: ini

    [datatypes]
    exposure = float
    n_points = int
    n_steps = int
    scan_type = str
    step_size = float
    x_sample = float
    y_sample = float
    z_sample = float
    r_sample = float

    [log_keys]
    exposure = [Exposure]
    n_points = [Points count]
    n_steps = [Steps count]
    scan_type = [Device]
    step_size = [Step size]
    x_sample = [X-SAM, SAM-X, SCAN-X]
    y_sample = [Y-SAM, SAM-Y, SCAN-Y]
    z_sample = [Z-SAM, SAM-Z, SCAN-Z]
    r_sample = [R-SAM, SAM-R, SCAN-R]

    [part_keys]
    exposure = Type: Method
    n_points = Type: Scan
    n_steps = Type: Scan
    scan_type = Type: Scan
    step_size = Type: Scan
    x_sample = Session logged attributes
    y_sample = Session logged attributes
    z_sample = Session logged attributes
    r_sample = Session logged attributes

:class:`LogContainer <cbclib.LogContainer>`
-------------------------------------------

Log data container class provides an interface to read Kamzik log files (:func:`read_logs <cbclib.LogContainer.read_logs>`), and convert
data to sample positions and sample orientations at each step of a scan stored into :class:`ScanSamples <cbclib.ScanSamples>` container.
Log container needs a :class:`cbclib.LogProtocol` log protocol. One can can either simulate the sample positions based on
initial and final read-outs and step sizes of the scan motors with :func:`simulate_translations <cbclib.LogContainer.simulate_translations>` or use
the motor positions of the whole scan with :func:`read_translations <cbclib.LogContainer.read_translations>`.

Contents
--------

.. toctree::
    :maxdepth: 1

    classes/log_protocol
    classes/log_container
