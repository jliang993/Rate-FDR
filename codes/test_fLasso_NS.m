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
para.c_gamma = 1/1;
%% NS - FDR :                         
% as different \gamma setting will lead to different z^\star, for each setting, we need to run
% the codes twice such that we can have ||z_k-z^\star||

para.val_gamma = @(k) 1* para.beta;
[xsol,z1,z2, ~, ~, ~] = func_nsFDR_fLasso(para, proxR1,proxR2,projV,gradF, 0,0);
[x,~,~, its, dk, ek] = func_nsFDR_fLasso(para, proxR1,proxR2,projV,gradF, xsol,[z1,z2]);

fprintf('\n');

% 1
para.val_gamma = @(k) (1 + 1./(k.^1.1))* para.beta;
[~,z1_1,z2_1, ~, ~, ~] = func_nsFDR_fLasso(para, proxR1,proxR2,projV,gradF, 0,0);
[x1,~,~, its1, dk1, ek1] = func_nsFDR_fLasso(para, proxR1,proxR2,projV,gradF, xsol,[z1_1,z2_1]);

fprintf('\n');

% 2
para.val_gamma = @(k) (1 + 1./(k.^2))* para.beta;
[~,z1_2,z2_2, ~, ~, ~] = func_nsFDR_fLasso(para, proxR1,proxR2,projV,gradF, 0,0);
[x2,~,~, its2, dk2, ek2] = func_nsFDR_fLasso(para, proxR1,proxR2,projV,gradF, xsol,[z1_2,z2_2]);

fprintf('\n');

% 3
para.val_gamma = @(k) (1 + .999.^k)* para.beta;
[~,z1_3,z2_3, ~, ~, ~] = func_nsFDR_fLasso(para, proxR1,proxR2,projV,gradF, 0,0);
[x3,~,~, its3, dk3, ek3] = func_nsFDR_fLasso(para, proxR1,proxR2,projV,gradF, xsol,[z1_3,z2_3]);

fprintf('\n');

% 4
para.val_gamma = @(k) (1 + .5.^k)* para.beta;
[~,z1_4,z2_4, ~, ~, ~] = func_nsFDR_fLasso(para, proxR1,proxR2,projV,gradF, 0,0);
[x4,~,~, its4, dk4, ek4] = func_nsFDR_fLasso(para, proxR1,proxR2,projV,gradF, xsol,[z1_4,z2_4]);
%% output Type: pdf or png
outputType = 'png';
%% plot ||zk - zsol||

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

figure(100), clf;
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters');
set(gcf,'paperposition',[0 0 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[0.6 0.3]);

p = semilogy(dk, 'r', 'LineWidth',linewidth);

hold on;
p1 = semilogy(dk1, 'm', 'LineWidth',linewidth);

p2 = semilogy(dk2, 'b', 'LineWidth',linewidth);
p3 = semilogy(dk3, 'Color',[.5,.5,.5], 'LineWidth',linewidth);

p4 = semilogy(dk4, 'k--', 'LineWidth',linewidth);

grid on;
axis([1, 1e3, 1e-10, 2*dk1(1)]);

% 
% 

ylabel({'$$\|z_k-z^\star\|$$'},...
    'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');
xlabel({'\vspace{-1.25mm}';'$$k$$'},...
    'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');

% 
% 


lg = legend([p, p1, p2, p3, p4], 'S-FDR', 'NS-FDR 1', 'NS-FDR 2', 'NS-FDR 3', 'NS-FDR 4');
set(lg,'FontSize', legendFontSize);
legend('boxoff');

epsname = sprintf('nsfdr_fLasso.%s', outputType);
if strcmp(outputType, 'png')
    print(epsname, '-dpng');
else
    print(epsname, '-dpdf');
end