import os

basedir = "bd_scheduler_logs"

for fn in os.listdir(basedir):
    if not os.path.isdir(os.path.join(basedir, fn)):
        continue  # Not a directory

    """ Renaming watts-strogatz models
    if "4_05" in fn:
        print(fn)
        first_half, params , second_half = fn.rpartition('4_05')
        begin, ba, num_nodes = first_half.rpartition("barabasi_albert")    
        new_name = begin + "ws" + num_nodes + params + second_half
        print(new_name)
        os.rename(os.path.join(basedir, fn),
                os.path.join(basedir, new_name))
        print("---------")
    """

    """ Attempt to delete files that don't fit naming convention
    if not fn.endswith("evens"):
        print(fn)
        os.rmdir(os.path.join(basedir, fn))
    """

    """ Renaming topo w/ seeds
    first_half, _, second_half = fn.rpartition('txt')
    #print(first_half)
    #print(second_half)
    new_name = first_half + "_0txt" + second_half
    print(new_name)
    os.rename(os.path.join(basedir, fn),
            os.path.join(basedir, new_name))
    """
    # print("---------")
