function [x, z1,z2, its, dk, ek, Dk, tk] = func_FDR_fLasso(para, proxR1,proxR2,projV,gradF, xsol,zsol,vsol)


fprintf(sprintf('performing FDR...\n'));
itsprint(sprintf('        step %08d: residual = %.3e...', 1,1), 1);


mu1 = para.mu1;
mu2 = para.mu2;
beta = para.beta;
n = para.n;

tol = para.tol;
maxits = para.maxits;

gamma = para.c_gamma*beta;
tau1 = mu1*gamma; 
tau2 = mu2*gamma;

z1 = zeros(n, 1);
z2 = z1;

v = [z1-z1; z2-z2];

x = projV(z1, z2);

dk = zeros(maxits, 1);
ek = zeros(maxits, 1);
tk = zeros(maxits, 1);

b = para.b;
A = para.A;
% Phi = @(u1, u2) mu1*norm(u1, 1) + mu2*norm(diff(u2), 1) + 1/2* norm(A*(u1/2+u2/2)-b, 2)^2;
Phi = @(u1, u2) para.mu1*norm(u1, 1) + para.mu2*norm(diff(u2), 1) ...
    + 1/2* norm(A*u1-b, 2)^2 + 1/2* norm(A*u2-b, 2)^2;

fsol = Phi(xsol, xsol);
Dk = zeros(maxits, 1);

its = 1;
while(its<maxits)
    
    z1_old = z1;
    z2_old = z2;
    
    v_old = v;
    
    grad = gamma* gradF(x);
    
    % soft-threshold
    u1 = proxR1(2*x-z1-grad, tau1);
    z1 = z1 + ( u1 - x );
    
    % TV
    u2 = proxR2(2*x-z2-grad, tau2);
    z2 = z2 + ( u2 - x );
    
    v = [z1-z1_old; z2-z2_old];
    
    % projection
    x = projV(z1, z2);
    
    % stop?
    res = norm([z1,z2]-[z1_old,z2_old], 'fro');
    if mod(its,1e1)==0; itsprint(sprintf('      step %08d: residual = %.3e...', its,res), its); end
    
    ek(its) = res;
    dk(its) = norm([z1,z2]-zsol, 'fro');
    
    tk(its) = ( (v')/norm(v) ) * ( (v_old) /norm(v_old) );
    
    % Bregman divergence
    % u_V = projV(u1, u2);
    
    Dk(its) = Phi(u1, u2) - fsol - (vsol')*([u1-xsol;u2-xsol]); %([u1-u_V; u2-u_V]);
    
    if (res<tol)||(res>1e10); break; end
    
    its = its + 1;
    
end
fprintf('\n');

dk = dk(1:its);
ek = ek(1:its);
tk = tk(1:its);
Dk = Dk(1:its);

% disp(norm(u1-xsol))
