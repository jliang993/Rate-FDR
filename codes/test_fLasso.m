clear all
close all
clc

warning off
addpath toolbox/
set(groot,'defaultLineLineWidth',1.5);
%%
m = 36;
n = 128;

% original object x0
x0 = zeros(n, 1);
x0(33:40,1) = [2;2;2; 2.5;2.5; 2;2;2];
x0(81:88,1) = [2;2; 2.5;2.5; 3;3; 3.5;3.5];

% linear operator
A = randn(m, prod(n)) /sqrt(m);

% observation
noise = 5e-2* randn(m,1);
b = A* x0(:) + noise;

gradF = @(x) (A')*(A*x - b);
proxR1 = @(x, t) wthresh(x, 's', t);
proxR2 = @(x, t) prox_tv1D(x, t);
projV = @(x1, x2) (x1 + x2) /2;
%%
para.tol = 1e-14;
para.maxits = 1e5;

para.n = n;

para.b = b;
para.A = A;

para.mu1 = 1/4;
para.mu2 = 1;

para.beta = 1/norm(A)^2;
para.c_gamma = 1/4;
%% FDR                          
[xsol,z1sol,z2sol] = func_FDR_fLasso(para, proxR1,proxR2,projV,gradF, 0,0, zeros(2*n,1));

gamma = para.c_gamma*para.beta;

zsol = [z1sol,z2sol];
vsol = [xsol-z1sol; xsol-z2sol] /gamma;

[x,z1,z2, its, dk, ek, fk, tk] = func_FDR_fLasso(para, proxR1,proxR2,projV,gradF, xsol,zsol,vsol);
%% Rate estimation                         
tol = 1e-10;
Id = eye(2*n);

% TV, PTj
I = find(abs(grad(x)) >= tol);
Ic= setdiff(1:n,I);

D = -diag(ones(n,1))+diag(ones(n-1,1),1);
D(end,:) = 0;

PTj_ = null(D(Ic,:));

PTj = PTj_*(PTj_');

% l1-norm, PTr
I = abs(x) >= tol;
PTr = diag(I);

% Hessian 
Hf = [(A')* A, 0*(A')* A; 0*(A')* A, (A')* A];

PV = [eye(n)/2, eye(n)/2; eye(n)/2, eye(n)/2];
Mr = [PTr, 0*PTr; 0*PTj, PTj];

M = Id + 2*Mr*PV - Mr - PV - gamma*Mr*(PV')*Hf*PV;
e_ = sort(abs(eig(M)));
e = e_(e_<1-1e-6);

rho = max(e);

k = 500;
dkT = 2e1* dk(k)* rho.^(1:length(dk));
%% output Type: pdf or png
outputType = 'png';
%% Plot Bregman divergence
linewidth = 1;

axesFontSize = 8;
labelFontSize = 12;
legendFontSize = 9;

resolution = 300; % output resolution
output_size = 300 *[10, 8]; % output size

figure(100), clf;
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[0.05 -0.10 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[0.625 0.5]);


p = loglog(max(fk(1:4e2), 1e-14), 'k', 'LineWidth',linewidth);
hold on,

p1 = loglog(1e3./(1:4e2), 'r', 'LineWidth',linewidth);

ax = gca;
ax.GridLineStyle = '--';

axis([1 4e2 1e-1 1e2]);

ytick = get(gca, 'yTick');
if length(ytick)>6
    set(gca, 'yTick', []);
    ytick = [1e-14 1e-10, 1e-6, 1e-2, 1e2];
    set(gca, 'yTick', ytick);
end

grid on;

ylabel({'$\inf_{1\leq i \leq k} D_{\Phi}^{v^\star}(u_{i})$'}, 'FontSize', labelFontSize,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
xlabel({'\vspace{-1.0mm}';'$k$'}, 'FontSize', labelFontSize,...
    'FontAngle', 'normal', 'Interpreter', 'latex');


lg = legend([p1, p], '$1/k$', '$\inf_{1\leq i \leq k} D_{\Phi}^{v^\star}(u_{i})$');
set(lg,'FontSize', legendFontSize);
set(lg, 'Location','SouthWest');
set(lg, 'Interpreter', 'latex');
% pos = get(lg, 'Position');
% set(lg, 'Position', [pos(1)+0.000, pos(2)+0.00, pos(3:4)]);
% pos_ = get(lg, 'Position');

legend('boxoff');

epsname = sprintf('fdr_fLasso_global.%s', outputType);
if strcmp(outputType, 'png')
    print(epsname, '-dpng');
else
    print(epsname, '-dpdf');
end
%% plot ||zk - zsol||

% k = 400;



linewidth = 1.25;
resolution = 300; % output resolution
output_size = 300 *[9, 7]; % output size

%
%

axesFontSize = 8;
labelFontSize = 10;
legendFontSize = 8;

% 
% 

figure(101), clf;
% figure('visible', 'off'),
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters');
set(gcf,'paperposition',[0 0 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[0.6 0.3]);

p1 = semilogy(1:length(dk),dk, 'k', 'LineWidth',linewidth);

hold on;
p2 = semilogy(k:length(dk),dkT(1:length(dk)-k+1),'r', 'LineWidth',linewidth);

grid on;
axis([1, its, 1e-12, dk(1)]);

% 
% 

ylabel({'$$\|z_k-z^\star\|$$'},...
    'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');
xlabel({'\vspace{-1.25mm}';'$$k$$'},...
    'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');

% 
% 


lg = legend([p2, p1], 'theoretical', 'practical');
set(lg,'FontSize', legendFontSize);
legend('boxoff');

epsname = sprintf('fdr_fLasso_local.%s', outputType);
if strcmp(outputType, 'png')
    print(epsname, '-dpng');
else
    print(epsname, '-dpdf');
end