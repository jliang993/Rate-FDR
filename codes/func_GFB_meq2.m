function [x, z1,z2, its, dk, ek] = func_GFB_meq2(para, proxR1,proxR2,gradF, zsol)


fprintf(sprintf('performing GFB...\n'));
itsprint(sprintf('        step %08d: residual = %.3e...', 1,1), 1);

w1 = 1/2;
w2 = 1 - w1;

mu1 = para.mu1;
mu2 = para.mu2;
beta = para.beta;
gamma = para.c_gamma*beta;
tau1 = mu1*gamma/w1; 
tau2 = mu2*gamma/w2;


tol = para.tol;
maxits = para.maxits;

projS = @(z1,z2) w1*z1 + w2*z2;

dk = zeros(maxits, 1);
ek = zeros(maxits, 1);

n = para.n;

z1 = para.b;
z2 = z1;
x = projS(z1, z2);

its = 1;
while(its<maxits)
    
    z1_old = z1;
    z2_old = z2;
    
    grad = gamma* gradF(x);
    
    % J_A, svt
    u1 = proxR1(2*x-z1-grad, tau1);
    z1 = z1 + ( u1 - x );
    
    % J_B, Non-negative Projection
    u2 = proxR2(2*x-z2-grad, tau2);
    z2 = z2 + ( u2 - x );
    
    % projection
    x = projS(z1, z2);
    
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
