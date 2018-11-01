clear all
close all
clc

warning off
addpath toolbox/
set(groot,'defaultLineLineWidth',1.5);
%%
n = [1, 1]* 50; % size of the mtx
r = round(max(n)/25); % rank of the mtx

% original x
xL = rand(n(1), r) *1;
xR = rand(r, n(2)) *2;
x0 = xL * xR; % matrix to be tested

% sub-sampling matrix
ratio = 0.5;
A = proj_mask(x0, ratio, 'p');

% observation
noise = 1e-2* randn(size(x0));
b = A .* x0 + noise;

gradF = @(x) A.*(x-b);
proxR = @(x, tau) svt(x, tau);
projV = @(x, tau) max(x, 0);
%%
para.tol = 1e-10;
para.maxits = 1e3;

para.b = b;

para.n = n;
para.mu1 = 1;
para.mu2 = 1;

para.beta = 1;
para.c_gamma = 1.0;
%% GFB
[~,z11,z21] = func_GFB_meq2(para, proxR,projV,gradF, 0);
[x1,z11,z21, its1, dk1, ek1] = func_GFB_meq2(para, proxR,projV,gradF, [z11,z21]);

fprintf('\n');
%% TOS
[~,z2] = func_TOS(para, proxR,projV,gradF, 0);
[x2,z2, its2, dk2, ek2] = func_TOS(para, proxR,projV,gradF, z2);

fprintf('\n');
%% rate estimation
gamma = para.c_gamma * para.beta;
A = diag(A(:)); 
rho = calc_eta_mc(x2, A, gamma, b(:), para.mu1);
%%
k = 30;
dkT = 2e1* dk2(k)* rho.^(1:length(dk2));
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

figure(101), clf;
% figure('visible', 'off'),
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters');
set(gcf,'paperposition',[0 0 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[0.6 0.3]);

p1 = semilogy(dk1, 'm', 'LineWidth',linewidth);

hold on;
p2 = semilogy(dk2,'k', 'LineWidth',linewidth);

p3 = semilogy(k:length(dk2),dkT(1:length(dk2)-k+1),'r', 'LineWidth',linewidth);

grid on;
axis([1, its2, 1e-10, 2*dk1(1)]);

%
%

ylabel({'$$\|z_k-z^\star\|$$'},...
    'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');
xlabel({'\vspace{-1.25mm}';'$$k$$'},...
    'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');

%
%


lg = legend([p1, p2, p3], 'GFB', 'TOS: practical', 'TOS: theoretical');
set(lg,'FontSize', legendFontSize);
legend('boxoff');

filename = sprintf('cmp_nMC_gfb_tos.png');
print(filename, '-dpng');
filename = sprintf('cmp_nMC_gfb_tos.pdf');
print(filename, '-dpdf');