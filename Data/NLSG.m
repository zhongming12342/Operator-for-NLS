clear
a = 0.2 + 0.6*rand(1500,1);
a(a==0.5) = 0.49;
x = linspace(-6,6,256);
t = linspace(-4,4,100);
[X,T] = meshgrid(x,t);
u = zeros(length(a),length(t),length(x));
for i = 1:length(a)
    b = sqrt(8*a(i)*(1-2*a(i)));
    w = 2*(sqrt(1-2*a(i)));
    A1 = 2*(1-2*a(i))*cosh(b*T)+1i*b*sinh(b*T);
    A2 = sqrt(2*a(i))*cos(w*X)-cosh(b*T);
    u(i,:,:) = exp(1i*T).*(1+A1./A2);
end
index = randi(length(a));
surf(x,t,abs(squeeze(u(index,:,:))))
shading interp
axis tight
colormap(jet)
u = permute(u,[1,3,2]);
% save NLSG a u x t
save('NLSG.mat', 'a', 'u', 'x', 't', '-v7');  % v7.2 ∏Ò Ω