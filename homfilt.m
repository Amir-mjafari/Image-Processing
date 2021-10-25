function res = homfilt(im,cutoff,order)

u = im2uint8(im);
u(find(u==0)) = 1;
l = log(double(u));
ft = fftshift(fft2(l));
f = hbutter(im,cutoff,order);
b = f .* ft;
ib = abs(ifft2(b));
res = exp(ib);
end