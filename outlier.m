function res = outlier (im,d)
f = [.125 .125 .125; .125 0 .125;.125 .125 .125]
imd = im2double(im);
imf = filter2(f,imd);
r = abs(imd-imf)-d > 0;
res = im2uint8(r.* imf + (1-r).*imd);
end