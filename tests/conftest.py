def pytest_addoption(parser):
    parser.addoption("--dir_path", type=str, default='/asap3/petra3/gpfs/p11/2021/data/11010570/raw',
                     help="Path to the experimental data")
    parser.addoption("--scan_num", type=int, default=209,
                     help="Scan number")
    parser.addoption("--roi", type=int, nargs=4, default=(1100, 3260, 1040, 3108),
                     help="Region of interest")

def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_dict = vars(metafunc.config.option)
    for attr in ('dir_path', 'scan_num', 'roi'):
        option_value = option_dict.get(attr)
        if attr in metafunc.fixturenames and option_value is not None:
            metafunc.parametrize(attr, [option_value])