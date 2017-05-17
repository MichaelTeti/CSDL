function data=read_data(filename);
% reads in the training and testing images for the small norb dataset

fid=fopen(filename,'r');
fread(fid,4,'uchar'); % result = [85 76 61 30], byte matrix(in base 16: [55 4C 3D 1E])
fread(fid,4,'uchar'); % result = [4 0 0 0], ndim = 4
fread(fid,4,'uchar'); % result = [236 94 0 0], dim0 = 24300 (=94*256+236)
fread(fid,4,'uchar'); % result = [2 0 0 0], dim1 = 2
fread(fid,4,'uchar'); % result = [96 0 0 0], dim2 = 96
fread(fid,4,'uchar'); % result = [96 0 0 0], dim3 = 96
a=fread(fid, 24300*96*96);
data=zeros(24300, 96, 96);
imsz=96*96;

for i=1:24300
    data(i, :, :)=imrotate(reshape(a(i*imsz-(imsz-1):i*imsz), 96, 96), 270);    
    
end 
