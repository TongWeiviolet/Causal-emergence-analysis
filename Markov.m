%% Numerical Experiments

% dumbbell Markov chain with 85 nodes and 5 groups

% determine the optimal classification number of the system for the
% occurrence of the causal emergence

clear
clc
%% Step1: Construct Markov chain

w1 = 25;
w2 = 25;
v1 = 15;
v2 = 15;
s = 5; 
r = 3;
DB = blkdiag(rand(w1),rand(v1),rand(s),rand(v2),rand(w2)); 

DB(w1-r:w1,w1+1:w1+1+r) = 1;                       
DB(w1+1:w1+1+r,w1-r:w1) = 1; 
DB(w1+v1,w1+v1+1) = 1; 
DB(w1+v1+1,w1+v1) = 1;
DB(w1+v1+s,w1+v1+s+1) = 1; 
DB(w1+v1+s+1,w1+v1+s) = 1;
DB(w1+v1+s+v2-r:w1+v1+s+v2,w1+v1+s+v2+1:w1+v1+s+v2+1+r) = 1; 
DB(w1+v1+s+v2+1:w1+v1+s+v2+1+r,w1+v1+s+v2-r:w1+v1+s+v2) = 1;

mc = dtmc(DB);

P0 = mc.P;
h = digraph(P0);

figure(1);
h1 = plot(h,'LineWidth',1);
highlight(h1,1:25,'NodeColor',[0.8500 0.3250 0.0980],'Marker','o','MarkerSize',8);
highlight(h1,26:40,'NodeColor',[0.9920 0.6000 0.1200],'Marker','s','MarkerSize',9);
highlight(h1,41:45,'NodeColor',[0.2560 0.6600 0.2560],'Marker','v','MarkerSize',10);
highlight(h1,46:60,'NodeColor',[0.3480 0.0160 0.0280],'Marker','^','MarkerSize',10);
highlight(h1,61:85,'NodeColor',[0.4940 0.1840 0.5560],'Marker','pentagram','MarkerSize',9);
h1.NodeLabel = {};
hold on;
a1 = plot(-10,-10,'o','Color',[0.8500 0.3250 0.0980],'MarkerFaceColor',[0.8500 0.3250 0.0980],'MarkerSize',9);
a2 = plot(-10,-10,'s','Color',[0.9920 0.6000 0.1200],'MarkerFaceColor',[0.9920 0.6000 0.1200],'MarkerSize',9);
a3 = plot(-10,-10,'v','Color',[0.2560 0.6600 0.2560],'MarkerFaceColor',[0.2560 0.6600 0.2560],'MarkerSize',9);
a4 = plot(-10,-10,'^','Color',[0.3480 0.0160 0.0280],'MarkerFaceColor',[0.3480 0.0160 0.0280],'MarkerSize',9);
a5 = plot(-10,-10,'pentagram','Color',[0.4940 0.1840 0.5560],'MarkerFaceColor',[0.4940 0.1840 0.5560],'MarkerSize',9);
legend([a1,a2,a3,a4,a5],'weight 1 with 25 nodes','weight 2 with 15 nodes',...
    'bar with 5 nodes','weight 3 with 15 nodes','weight 4 with 25 nodes',...
    'Location','southeast','FontName','Calibri','FontSize',13,'FontWeight','bold');
xlim([-5 5]);
ylim([-5 5]);

%% Step 2: Construct time-series data

P = mc.P;
P = P';
N = size(P,1);
T = 200;
x = zeros(N,T);
x(:,1) = rand(N,1);
x(:,1) = x(:,1)./sum(x(:,1));
for i = 2:T
    x(:,i) = P * x(:,i-1);
end

%% Step 3: Generate a DPTM

Xs = x(:,21:T-2);
Xi = x(:,22:T-1) - x(:,21:T-2);
Ys = x(:,22:T-1);
Yi = x(:,23:T) - x(:,22:T-1);

cvx_begin 
    variable Ps(N,N) nonnegative
    minimize (norm(Ys - Ps * Xs,'fro'))
    subject to
        ones(1,N) * Ps == ones(1,N);
cvx_end

cvx_begin 
    variable Pi(N,N)
    minimize (norm(Yi - Ps * Xi - Pi * Xs,'fro'))
    subject to
        ones(1,N) * Pi == zeros(1,N);
        Pi(Ps < 1e-9) >= 0;
cvx_end

figure(2);
t1 = tiledlayout(1,3,'TileSpacing','Compact');

nexttile
imagesc(P);
title('Theoretical TPM');
axis square;
colorbar;

nexttile
imagesc(Ps);
title('Simulated DTPM (standard part)');
axis square;
colorbar;

nexttile
imagesc(Pi);
title('Simulated DTPM (infinitesimal part)');
axis square;
colorbar;

%% Step 4: Calculate the dual-valued Ky Fan p-k-norm

q = 1:0.3:1.9;
m = size(q,2);
ss1 = cell(m,1);
ss2 = cell(m,1);
Pd(:,:,1) = Ps;
Pd(:,:,2) = Pi;
for k = 1:N
    for p = 1:m
        n = DKFnorm(Pd,q(p),k);
        ss1{p}(k,1) = n(1,1);
        ss2{p}(k,1) = n(1,2);
    end
end

figure(3);
tiledlayout(2,2,'TileSpacing','Compact');
nexttile
plot(ss1{1},'-','Color',[1 0.27 0.23],'Marker','o','MarkerSize',11,...
    'MarkerFaceColor',[0.97,0.67,0.51],LineWidth=2);
xlabel('k');
title('p=1: standard part',FontWeight = 'bold');
ax = gca;
ax.FontName = 'Calibri';
ax.FontSize = 26;
ax.FontWeight = 'bold'; 

nexttile(3)
plot(ss2{1},'--','Color',[0 0.45 0.74],'Marker','diamond','MarkerSize',11,...
    'MarkerFaceColor',[0.47,0.82,0.97],LineStyle=':',LineWidth=2);
xlabel('k');
xline(5,'Color',[0.39,0.83,0.07],'LineWidth',4,'Alpha',0.5,...
    'Label','k=5', 'FontName','Calibri','FontSize',26.4,...
    'FontWeight','bold','LabelColor',[0.47,0.67,0.19],...
    'LabelVerticalAlignment','bottom','LabelOrientation','horizontal');
title('p=1: infinitesimal part',FontWeight = 'bold');
ax = gca;
ax.FontName = 'Calibri';
ax.FontSize = 26;
ax.FontWeight = 'bold'; 

nexttile(2)
plot(ss1{2},'-','Color',[1 0.27 0.23],'Marker','o','MarkerSize',11,...
    'MarkerFaceColor',[0.97,0.67,0.51],LineWidth=2);
xlabel('k');
title('p=1.3: standard part',FontWeight = 'bold');
ax = gca;
ax.FontName = 'Calibri';
ax.FontSize = 26;
ax.FontWeight = 'bold'; 

nexttile
plot(ss2{2},'--','Color',[0 0.45 0.74],'Marker','diamond','MarkerSize',11,...
    'MarkerFaceColor',[0.47,0.82,0.97],LineStyle=':',LineWidth=2);
xlabel('k');
xline(5,'Color',[0.39,0.83,0.07],'LineWidth',4,'Alpha',0.5,...
    'Label','k=5','FontName','Calibri','FontSize',26.4,...
    'FontWeight','bold','LabelColor',[0.47,0.67,0.19],...
    'LabelVerticalAlignment','bottom','LabelOrientation','horizontal');
title('p=1.3: infinitesimal part',FontWeight = 'bold');
ax = gca;
ax.FontName = 'Calibri';
ax.FontSize = 26;
ax.FontWeight = 'bold'; 

figure(4);
tiledlayout(2,2,'TileSpacing','Compact');
nexttile
plot(ss1{3},'-','Color',[1 0.27 0.23],'Marker','o','MarkerSize',11,...
    'MarkerFaceColor',[0.97,0.67,0.51],LineWidth=2);
xlabel('k');
title('p=1.6: standard part',FontWeight = 'bold');
ax = gca;
ax.FontName = 'Calibri';
ax.FontSize = 26;
ax.FontWeight = 'bold'; 

nexttile(3)
plot(ss2{3},'--','Color',[0 0.45 0.74],'Marker','diamond','MarkerSize',11,...
    'MarkerFaceColor',[0.47,0.82,0.97],LineStyle=':',LineWidth=2);
xlabel('k');
xline(5,'Color',[0.39,0.83,0.07],'LineWidth',4,'Alpha',0.5,...
    'Label','k=5','FontName','Calibri','FontSize',26.4,...
    'FontWeight','bold','LabelColor',[0.47,0.67,0.19],...
    'LabelVerticalAlignment','bottom','LabelOrientation','horizontal');
title('p=1.6: infinitesimal part',FontWeight = 'bold');
ax = gca;
ax.FontName = 'Calibri';
ax.FontSize = 26;
ax.FontWeight = 'bold'; 

nexttile(2)
plot(ss1{4},'-','Color',[1 0.27 0.23],'Marker','o','MarkerSize',11,...
    'MarkerFaceColor',[0.97,0.67,0.51],LineWidth=2);
xlabel('k');
title('p=1.9: standard part',FontWeight = 'bold');
ax = gca;
ax.FontName = 'Calibri';
ax.FontSize = 26;
ax.FontWeight = 'bold'; 

nexttile
plot(ss2{4},'--','Color',[0 0.45 0.74],'Marker','diamond','MarkerSize',11,...
    'MarkerFaceColor',[0.47,0.82,0.97],LineStyle=':',LineWidth=2);
xlabel('k');
xline(5,'Color',[0.39,0.83,0.07],'LineWidth',4,'Alpha',0.5,...
    'Label','k=5','FontName','Calibri','FontSize',26.4,...
    'FontWeight','bold','LabelColor',[0.47,0.67,0.19],...
    'LabelVerticalAlignment','bottom','LabelOrientation','horizontal');
title('p=1.9: infinitesimal part',FontWeight = 'bold');
ax = gca;
ax.FontName = 'Calibri';
ax.FontSize = 26;
ax.FontWeight = 'bold'; 

%% Step 5: Cluster with the infinitesimal part

[Ud,Sd,Vd] = dualsvd(Pd);

index = 5;
Pst = Sd(1:index,1:index,1) * Vd(:,1:index,1)';
Pin = Sd(1:index,1:index,1) * Vd(:,1:index,2)' +...
Sd(1:index,1:index,2) * Vd(:,1:index,1)';
Q1 = [Pst;Pin];
% cluster "index" classes 
[id,~] = kmeans(Q1',index);
phi1 = zeros(N,index);
for i = 1:N
    phi1(i,id(i)) = 1;
end
F1 = phi1'*Ps*phi1;
% reduced TPM
G1 = F1./sum(F1);

figure(5);
imagesc(G1);          
colorbar();    
colormap(sky);
for i=1:5 
    for j=1:5
        if G1(i,j)<1e-4
            text(j,i,sprintf('0'),'FontName','Calibri','FontSize',16,...
                'FontWeight','bold','HorizontalAlignment','center');
        else
            text(j,i,sprintf('%.4f',G1(i,j)),'FontName','Calibri','FontSize',16,...
                'FontWeight','bold','HorizontalAlignment','center');
        end
    end
end
title('Cluster with the infinitesimal part',FontWeight = 'bold');
axis square
ax = gca;
ax.FontName = 'Calibri';
ax.FontSize = 16;
ax.FontWeight = 'bold';

figure(6);
imagesc(phi1');
colormap(sky);
xlabel('Nodes');
ylabel('Classes');
ax = gca;
ax.FontName = 'Calibri';
ax.FontSize = 16;
ax.FontWeight = 'bold';

%% Step 6: Cluster without the infinitesimal part

index = 5;
Q2 = Sd(1:index,1:index,1) * Vd(:,1:index,1)';
[id,~] = kmeans(Q2',index);
phi2 = zeros(N,index);
for i = 1:N
    phi2(i,id(i)) = 1;
end
F2 = phi2'*Ps*phi2;
% reduced TPM
G2 = F2./sum(F2);

figure(7);
imagesc(G2);          
colorbar();    
colormap(sky);
for i=1:5 
    for j=1:5
        if G2(i,j)<1e-4
            text(j,i,sprintf('0'),'FontName','Calibri','FontSize',16,...
                'FontWeight','bold','HorizontalAlignment','center');
        else
            text(j,i,sprintf('%.4f',G2(i,j)),'FontName','Calibri','FontSize',16,...
                'FontWeight','bold','HorizontalAlignment','center');
        end
    end
end
title('Cluster without the infinitesimal part',FontWeight = 'bold');
axis square
ax = gca;
ax.FontName = 'Calibri';
ax.FontSize = 16;
ax.FontWeight = 'bold';

figure(8);
imagesc(phi2');
colormap(sky);
xlabel('Nodes');
ylabel('Classes');
ax = gca;
ax.FontName = 'Calibri';
ax.FontSize = 16;
ax.FontWeight = 'bold';
