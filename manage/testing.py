import inspect


def do_test(obj):
    object_tested = obj()
    methods = inspect.getmembers(object_tested, inspect.ismethod)
    for i in range(len(methods)):
        test_name = methods[i][0]
        try:
            eval(f'object_tested.{test_name}()')
        finally:
            print(f'Test {i+1}/{len(methods)} Passed.')
