function eta = calc_eta_mc(x, A, gamma, f, mu)

[n1, n2] = size(x);
r_ = rank(x);

[UF,S,VF] = svd(x);
U  = UF(:, 1:r_);
V  = VF(:, 1:r_);
S  = diag(S);
S(r_+1:end) = 0;
xp = U*diag(1./S(1:r_))*(V');
gFx = reshape((A')*(A*x(:)-f), [n1, n2]);

PT = @(h) U*(U')*h + h*V*(V') - U*(U')*h*V*(V');
PS = @(h) h - PT(h);
hessF = @(h) reshape((A')*A*h(:), [n1, n2]);
G  = @(h) h - gamma*PT(hessF(h));
weig = @(h,v) v*(h')*xp + xp*(h')*v;
Q = @(h) gamma*(PT(hessJ(h, UF,VF,S, mu)) + weig(h,PS(gFx)));
afun = @(h) reshape(reshape(h, [n1, n2])+Q(PT(reshape(h, [n1, n2]))),[n2^2 1]);

h = ones(n1, n2);
for i=1:1e3
    % i
    [hnew,~] = gmres(afun,reshape(G(PT(h)),[n2^2 1]));
    hnew = reshape(hnew, [n1, n2]);
    
    eta = trace(hnew'*h)/norm(h,'fro')^2;%norm(h,'fro');
    
    h = hnew/norm(h,'fro');
end