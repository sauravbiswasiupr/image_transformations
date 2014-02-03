import doctest, sys, pkgutil

def runTests():
    import ift6266
    for (_, name, ispkg) in pkgutil.walk_packages(ift6266.__path__, ift6266.__name__+'.'):
        if not ispkg:
            if name.startswith('ift6266.scripts.') or \
               name.startswith('ift6266.data_generation.transformations.pycaptcha.') or \
               name in ['ift6266.test',
                        'ift6266.data_generation.transformations.testmod',
                        'ift6266.data_generation.transformations.gimp_script']:
                continue
            test(name)

def test(name):
    import ift6266
    predefs = ift6266.__dict__
    options = doctest.ELLIPSIS or doctest.DONT_ACCEPT_TRUE_FOR_1
    print "Testing:", name
    __import__(name)
    doctest.testmod(sys.modules[name], extraglobs=predefs, optionflags=options)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        for mod in sys.argv[1:]:
            if mod.endswith('.py'):
                mod = mod[:-3]
            test(mod)
    else:
        runTests()
