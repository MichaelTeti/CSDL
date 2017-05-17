function labels=read_labels(filename)

fid=fopen(filename,'r');
fread(fid,4,'uchar'); % result = [84 76 61 30], int matrix (54 4C 3D 1E)
fread(fid,4,'uchar'); % result = [1 0 0 0], ndim = 1
fread(fid,4,'uchar'); % result = [236 94 0 0], dim0 = 24300
fread(fid,4,'uchar'); % result = [1 0 0 0] (ignore this integer)
fread(fid,4,'uchar'); % result = [1 0 0 0] (ignore this integer)
labels=fread(fid,24300,'int'); % result = [0 1 2 3 4 0 1 2 3 4] (only on little-endian)



end
