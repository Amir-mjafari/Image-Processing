function out = cconv(a,b)
if length (a) ~= length(b)
    error ('vectors must be the same length')
end
a1 = length (a);
t = conv([a a],b);
out = t(a1+1:2*a1);
end

