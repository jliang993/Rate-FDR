function [x,z, its, dk, ek] = func_TOS(para, proxR,projV,gradF, zsol)

fprintf(sprintf('performing TOS...\n'));
itsprint(sprintf('        step %08d: residual = %.3e...', 1,1), 1);

mu1 = para.mu1;
mu2 = para.mu2;
beta = para.beta;
gamma = para.c_gamma*beta;
tau1 = mu1*gamma;
tau2 = mu2*gamma;


tol = para.tol;
maxits = para.maxits;


n = para.n;
z = para.b;
x = projV(z, tau2);

dk = zeros(maxits, 1);
ek = zeros(maxits, 1);


its = 1;
while(its<maxits)
    
    z_old = z;
    
    grad = gamma* gradF(x);
    
    % proxR
    u = proxR(2*x-z-grad, tau1);
    z = z + ( u - x );
    
    % proxJ
    x = projV(z, tau2);
    
    % stop?
    res = norm(z-z_old, 'fro');
    if mod(its,1e1)==0; itsprint(sprintf('      step %08d: residual = %.3e...', its,res), its); end
    
    ek(its) = res;
    dk(its) = norm(z-zsol, 'fro');
    
    if (res<tol)||(res>1e10); break; end
    
    its = its + 1;
    
end
fprintf('\n');

ek = ek(1:its-1);