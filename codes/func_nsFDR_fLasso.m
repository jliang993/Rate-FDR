function [x, z1,z2, its, dk, ek] = func_nsFDR_fLasso(para, proxR1,proxR2,projV,gradF, xsol,zsol)


fprintf(sprintf('performing NS-FDR...\n'));
itsprint(sprintf('        step %08d: residual = %.3e...', 1,1), 1);


mu1 = para.mu1;
mu2 = para.mu2;
beta = para.beta;
n = para.n;

tol = para.tol;
maxits = para.maxits;

gamma = para.c_gamma*beta;
% tau1 = mu1*gamma; 
% tau2 = mu2*gamma;

z1 = zeros(n, 1);
z2 = z1;

x = projV(z1, z2);

dk = zeros(maxits, 1);
ek = zeros(maxits, 1);

val_gamma = para.val_gamma;


its = 1;
while(its<maxits)
    
    z1_old = z1;
    z2_old = z2;
    
    gamma = val_gamma(its);
    
    grad = gamma* gradF(x);
    
    % J_A, soft-threshold
    u1 = proxR1(2*x-z1-grad, mu1*gamma);
    z1 = z1 + ( u1 - x );
    
    % J_B, Non-negative Projection
    u2 = proxR2(2*x-z2-grad, mu2*gamma);
    z2 = z2 + ( u2 - x );
    
    % projection
    x = projV(z1, z2);
    
    % stop?
    res = norm([z1,z2]-[z1_old,z2_old], 'fro');
    if mod(its,1e1)==0; itsprint(sprintf('      step %08d: residual = %.3e...', its,res), its); end
    
    ek(its) = res;
    dk(its) = norm([z1,z2]-zsol, 'fro');
    
    
    if (res<tol)||(res>1e10); break; end
    
    its = its + 1;
    
end
fprintf('\n');

dk = dk(1:its);
ek = ek(1:its);

% disp(norm(u1-xsol))
