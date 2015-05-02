# switches
params.DO_PREPROCESSING = 1
params.DO_RESIZE = 0
params.DO_GRAYLEVEL = 1
params.DO_PATCHNORMALIZATION = 1
params.DO_SAVE_PREPROCESSED_IMG = 0
params.DO_DIFF_MATRIX = 1
params.DO_CONTRAST_ENHANCEMENT = 1
params.DO_FIND_MATCHES = 1


# parameters for preprocessing
params.downsample.size = mcat([32, 64])# height, width
params.downsample.method = mstring('lanczos3')
params.normalization.sideLength = 8
params.normalization.mode = 1


# parameters regarding the matching between images
params.matching.ds = 10
params.matching.Rrecent = 5
params.matching.vmin = 0.8
params.matching.vskip = 0.1
params.matching.vmax = 1.2
params.matching.Rwindow = 10
params.matching.save = 1
params.matching.load = 1

# parameters for contrast enhancement on difference matrix  
params.contrastEnhancement.R = 10

# load old results or re-calculate? save results?
params.differenceMatrix.save = 1
params.differenceMatrix.load = 1

params.contrastEnhanced.save = 1
params.contrastEnhanced.load = 1

# suffix appended on files containing the results
params.saveSuffix = mstring('')

end

@mfunction("results")
def doDifferenceMatrix(results=None, params=None):

    filename = sprintf(mstring('%s/difference-%s-%s%s.mat'), params.savePath, params.dataset(1).saveFile, params.dataset(2).saveFile, params.saveSuffix)

    if params.differenceMatrix.load and exist(filename, mstring('file')):
        display(sprintf(mstring('Loading image difference matrix from file %s ...'), filename))

        d = load(filename)
        results.D = d.D
    else:
        if length(results.dataset) < 2:
            display(mstring('Error: Cannot calculate difference matrix with less than 2 datasets.'))
            return
        end

        display(mstring('Calculating image difference matrix ...'))
        #      h_waitbar = waitbar(0,'Calculating image difference matrix');

        n = size(results.dataset(1).preprocessing, 2)
        m = size(results.dataset(2).preprocessing, 2)

        D = zeros(n, m, mstring('single'))

        # parfor!
        for i in mslice[1:n]:
            d = results.dataset(2).preprocessing - repmat(results.dataset(1).preprocessing(mslice[:], i), 1, m)
            D(i, mslice[:]).lvalue = sum(abs(d)) / n
            #waitbar(i/n);
        end
        results.D = D

        # save it
        if params.differenceMatrix.save:
            save(filename, mstring('D'))
        end

        #   close(h_waitbar);
    end


end


@mfunction("results")
def doFindMatches(results=None, params=None):

    filename = sprintf(mstring('%s/matches-%s-%s%s.mat'), params.savePath, params.dataset(1).saveFile, params.dataset(2).saveFile, params.saveSuffix)

    if params.matching.load and exist(filename, mstring('file')):
        display(sprintf(mstring('Loading matchings from file %s ...'), filename))
        m = load(filename)
        results.matches = m.matches
    else:

        matches = NaN(size(results.DD, 2), 2)

        display(mstring('Searching for matching images ...'))
        # h_waitbar = waitbar(0, 'Searching for matching images.');

        # make sure ds is dividable by two
        params.matching.ds = params.matching.ds + mod(params.matching.ds, 2)

        DD = results.DD
        #parfor!
        for N in mslice[params.matching.ds / 2 + 1:size(results.DD, 2) - params.matching.ds / 2]:
            matches(N, mslice[:]).lvalue = findSingleMatch(DD, N, params)
            #   waitbar(N / size(results.DD,2), h_waitbar);
        end

        # save it
        if params.matching.save:
            save(filename, mstring('matches'))
        end

        results.matches = matches
    end
end


#%
@mfunction("match")
def findSingleMatch(DD=None, N=None, params=None):


    # We shall search for matches using velocities between
    # params.matching.vmin and params.matching.vmax.
    # However, not every vskip may be neccessary to check. So we first find
    # out, which v leads to different trajectories:

    move_min = params.matching.vmin * params.matching.ds
    move_max = params.matching.vmax * params.matching.ds

    move = mslice[move_min:move_max]
    v = move / params.matching.ds

    idx_add = repmat(mcat([mslice[0:params.matching.ds]]), size(v, 2), 1)
    # idx_add  = floor(idx_add.*v);
    idx_add = floor(idx_add *elmul* repmat(v.cT, 1, length(idx_add)))

    # this is where our trajectory starts
    n_start = N - params.matching.ds / 2
    x = repmat(mcat([mslice[n_start:n_start + params.matching.ds]]), length(v), 1)


    score = zeros(1, size(DD, 1))

    # add a line of inf costs so that we penalize running out of data
    DD = mcat([DD, OMPCSEMI, inf(1, size(DD, 2))])

    y_max = size(DD, 1)
    xx = (x - 1) * y_max

    for s in mslice[1:size(DD, 1)]:
        y = min(idx_add + s, y_max)
        idx = xx + y
        score(s).lvalue = min(sum(DD(idx), 2))
    end


    # find min score and 2nd smallest score outside of a window
    # around the minimum 

    [min_value, min_idx] = min(score)
    window = mslice[max(1, min_idx - params.matching.Rwindow / 2):min(length(score), min_idx + params.matching.Rwindow / 2)]
    not_window = setxor(mslice[1:length(score)], window)
    min_value_2nd = min(score(not_window))

    match = mcat([min_idx + params.matching.ds / 2, OMPCSEMI, min_value / min_value_2nd])
end



@mfunction("results")
def doPreprocessing(params=None):
    for i in mslice[1:length(params.dataset)]:

        # shall we just load it?
        filename = sprintf(mstring('%s/preprocessing-%s%s.mat'), params.dataset(i).savePath, params.dataset(i).saveFile, params.saveSuffix)
        if params.dataset(i).preprocessing.load and exist(filename, mstring('file')):
            r = load(filename)
            display(sprintf(mstring('Loading file %s ...'), filename))
            results.dataset(i).preprocessing.lvalue = r.results_preprocessing
        else:
            # or shall we actually calculate it?
            p = params
            p.dataset = params.dataset(i)
            results.dataset(i).preprocessing.lvalue = single(preprocessing(p))

            if params.dataset(i).preprocessing.save:
                results_preprocessing = single(results.dataset(i).preprocessing)
                save(filename, mstring('results_preprocessing'))
            end
        end
    end
end



#%
@mfunction("images")
def preprocessing(params=None):


    display(sprintf(mstring('Preprocessing dataset %s, indices %d - %d ...'), params.dataset.name, params.dataset.imageIndices(1), params.dataset.imageIndices(end)))
    # h_waitbar = waitbar(0,sprintf('Preprocessing dataset %s, indices %d - %d ...', params.dataset.name, params.dataset.imageIndices(1), params.dataset.imageIndices(end)));

    # allocate memory for all the processed images
    n = length(params.dataset.imageIndices)
    m = params.downsample.size(1) * params.downsample.size(2)

    if not isempty(params.dataset.crop):
        c = params.dataset.crop
        m = (c(3) - c(1) + 1) * (c(4) - c(2) + 1)
    end

    images = zeros(m, n, mstring('uint8'))
    j = 1

    # for every image ....
    for i in params.dataset.imageIndices:
        filename = sprintf(mstring('%s/%s%05d%s%s'), params.dataset.imagePath, params.dataset.prefix, i, params.dataset.suffix, params.dataset.extension)

        img = imread(filename)

        # convert to grayscale
        if params.DO_GRAYLEVEL:
            img = rgb2gray(img)
        end


        # resize the image
        if params.DO_RESIZE:
            img = imresize(img, params.downsample.size, params.downsample.method)
        end


        # crop the image if necessary
        if not isempty(params.dataset.crop):
            img = img(mslice[params.dataset.crop(2):params.dataset.crop(4)], mslice[params.dataset.crop(1):params.dataset.crop(3)])
        end

        # do patch normalization
        if params.DO_PATCHNORMALIZATION:
            img = patchNormalize(img, params)
        end


        # shall we save the result?
        if params.DO_SAVE_PREPROCESSED_IMG:

        end

        images(mslice[:], j).lvalue = img(mslice[:])
        j = j + 1

        # waitbar((i-params.dataset.imageIndices(1)) / (params.dataset.imageIndices(end)-params.dataset.imageIndices(1)));

    end

    # close(h_waitbar);

end


@mfunction("img")
def patchNormalize(img=None, params=None):
    s = params.normalization.sideLength

    n = mslice[1:s:size(img, 1) + 1]
    m = mslice[1:s:size(img, 2) + 1]

    for i in mslice[1:length(n) - 1]:
        for j in mslice[1:length(m) - 1]:
            p = img(mslice[n(i):n(i + 1) - 1], mslice[m(j):m(j + 1) - 1])

            pp = p(mslice[:])

            if params.normalization.mode != 0:
                pp = double(pp)
                img(mslice[n(i):n(i + 1) - 1], mslice[m(j):m(j + 1) - 1]).lvalue = 127 + reshape(round((pp - mean(pp)) / std(pp)), s, s)
            else:
                f = 255.0 / double(max(pp) - min(pp))
                img(mslice[n(i):n(i + 1) - 1], mslice[m(j):m(j + 1) - 1]).lvalue = round(f * (p - min(pp)))
            end


        end
    end

end


@mfunction("results")
def doContrastEnhancement(results=None, params=None):

    filename = sprintf(mstring('%s/differenceEnhanced-%s-%s%s.mat'), params.savePath, params.dataset(1).saveFile, params.dataset(2).saveFile, params.saveSuffix)

    if params.contrastEnhanced.load and exist(filename, mstring('file')):
        display(sprintf(mstring('Loading contrast-enhanced image distance matrix from file %s ...'), filename))
        dd = load(filename)
        results.DD = dd.DD

    else:

        display(mstring('Performing local contrast enhancement on difference matrix ...'))
        #    h_waitbar = waitbar(0,'Local contrast enhancement on difference matrix');

        DD = zeros(size(results.D), mstring('single'))

        D = results.D
        # parfor!
        for i in mslice[1:size(results.D, 1)]:
            a = max(1, i - params.contrastEnhancement.R / 2)
            b = min(size(D, 1), i + params.contrastEnhancement.R / 2)
            v = D(mslice[a:b], mslice[:])
            DD(i, mslice[:]).lvalue = (D(i, mslice[:]) - mean(v)) /eldiv/ std(v)
            # waitbar(i/size(results.D, 1));                
        end

        # let the minimum distance be 0
        results.DD = DD - min(min(DD))

        # save it?
        if params.contrastEnhanced.save:
            DD = results.DD
            save(filename, mstring('DD'))
        end

        #      close(h_waitbar);

    end

end



@mfunction("results")
def openSeqSLAM(params=None):

    results = mcat([])

    # try to allocate 3 CPU cores (adjust to your machine) for parallel
    # processing
        try:
        if matlabpool(mstring('size')) == 0:
            matlabpool(mstring('3'))
        end