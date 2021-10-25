function out = lbutter(im,d,n)
height = size (im,1);
width = size(im,2);
[x,y] =  meshgrid(-floor(width/2):floor((width-1)/2),-floor(height/2):floor((height-1)/2));
out = 1./(1+(sqrt(2)-1)*((x.^2+y.^2)/d^2).^n);
end