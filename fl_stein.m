function y = fl_stein(x)
%floyd-steinberg error diffusion to an image x which is unit8
height = size(x,1);
width = size(x,2);
y = zeros(height,width);
z = zeros(height+2,width+2);
z(2:height+1,2:width+1) = x;
for i = 2:height+1
    for j = 2:width+1
        if z(i,j) < 128
            y(i-1,j-1) = 0;
            e = z(i,j);
        else
            y(i-1,j-1) = 255;
            e = z(i,j)-255;
        end
        z(i,j+1) = z(i,j+1) + 7*e/16;
        z(i+1,j-1) = z(i+1,j-1) + 3*e/16;
        z(i+1,j) = z(i+1,j) + 5*e/16;
        z(i+1,j+1) = z(i+1,j+1) + e/16;
    end
end
