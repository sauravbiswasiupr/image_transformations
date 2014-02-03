from ift6266.deep.convolutional_dae.scdae import *

class dumb(object):
    COMPLETE = None
    def save(self):
        pass

def go(state, channel):
    from ift6266 import datasets
    from ift6266.deep.convolutional_dae.sgd_opt import sgd_opt
    import pylearn, theano, ift6266
    import pylearn.version
    import sys

    # params: bsize, pretrain_lr, train_lr, nfilts1, nfilts2, nftils3, nfilts4
    #         pretrain_rounds, noise, mlp_sz

    pylearn.version.record_versions(state, [theano, ift6266, pylearn])
    # TODO: maybe record pynnet version?
    channel.save()

    dset = datasets.nist_P07()

    nfilts = []
    fsizes = []
    if state.nfilts1 != 0:
        nfilts.append(state.nfilts1)
        fsizes.append((5,5))
        if state.nfilts2 != 0:
            nfilts.append(state.nfilts2)
            fsizes.append((3,3))
            if state.nfilts3 != 0:
                nfilts.append(state.nfilts3)
                fsizes.append((3,3))
                if state.nfilts4 != 0:
                    nfilts.append(state.nfilts4)
                    fsizes.append((2,2))

    subs = [(2,2)]*len(nfilts)
    noise = [state.noise]*len(nfilts)

    pretrain_funcs, trainf, evalf, net = build_funcs(
        img_size=(32, 32),
        batch_size=state.bsize,
        filter_sizes=fsizes,
        num_filters=nfilts,
        subs=subs,
        noise=noise,
        mlp_sizes=[state.mlp_sz],
        out_size=62,
        dtype=numpy.float32,
        pretrain_lr=state.pretrain_lr,
        train_lr=state.train_lr)

    t_it = repeat_itf(dset.train, state.bsize)
    pretrain_fs, train, valid, test = massage_funcs(
        t_it, t_it, dset, state.bsize, 
        pretrain_funcs, trainf,evalf)

    series = create_series()

    print "pretraining ..."
    sys.stdout.flush()
    do_pretrain(pretrain_fs, state.pretrain_rounds, series['recons_error'])

    print "training ..."
    sys.stdout.flush()
    best_valid, test_score = sgd_opt(train, valid, test,
                                     training_epochs=800000, patience=2000,
                                     patience_increase=2.,
                                     improvement_threshold=0.995,
                                     validation_frequency=500,
                                     series=series, net=net)
    state.best_valid = best_valid
    state.test_score = test_score
    channel.save()
    return channel.COMPLETE

if __name__ == '__main__':
    st = dumb()
    st.bsize = 100
    st.pretrain_lr = 0.01
    st.train_lr = 0.1
    st.nfilts1 = 4
    st.nfilts2 = 4
    st.nfilts3 = 0
    st.pretrain_rounds = 500
    st.noise=0.2
    st.mlp_sz = 500
    go(st, dumb())
