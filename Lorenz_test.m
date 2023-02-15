clear 
clc
dim = 7;tau = 1;u =1;alpha = 1.01;
i = 1;
j = 200;
m = 1000;
x0 = [0,0,0.4,0.3,0.3,0.3];
tspan = [0.1:0.1:300];
texy = [];
teyx = [];
texy_threshold = [];
teyx_threshold = [];
for c = 0:0.1:5
    c
    [t,y] = ode45(@(t,x) Lorenz(t,x,c),tspan,x0);
    driver = y(:,1:3);
    response = y(:,4:6);
    x2 = driver(1001:length(driver),2)';
    y2 = response(1001:length(driver),2)';
    x2 = (x2-mean(x2))/std(x2);
    y2 = (y2-mean(y2))/std(y2);
% dim = Cao(y2,2)
    texy = [texy,kRTE2(x2,y2,dim,tau,u,alpha)]
    teyx = [teyx,kRTE2(y2,x2,dim,tau,u,alpha)]
    texy_threshold = [texy_threshold,p_test(x2,y2,dim,tau,u,alpha,i,j,m,'kte')]
    teyx_threshold = [teyx_threshold,p_test(y2,x2,dim,tau,u,alpha,i,j,m,'kte')];
    
end

deltatexy = texy-teyx;
save('Loren.mat','y')
save('te.mat','texy','teyx')
save('te_threshold.mat','texy_threshold','teyx_threshold')

t = 0:0.1:5;
yyy = zeros(1,length(t));

figure('Color','white');
plot(t,texy,'r-*',t,teyx,'k-*',t,deltatexy,'b',t,yyy,'--')
legend('TE_M(x \rightarrow y)','TE_M(y \rightarrow x)','\Delta TE_M(x \rightarrow y)')
xlabel('Coupling C ')
title('Causality measures for Rössler system driving Lorenz system')
figure('Color','white')
plot(t,texy,'r-.',t,texy_threshold,'k-*')
legend('TE_M(x \rightarrow y)','Threshold(x \rightarrow y)')
xlabel('Coupling C ')
title('Significance Threshold')
figure(3)
plot(t,teyx_threshold,'r.',t,teyx,'ko')

% figure('Color','white')
% subplot(1,2,1)
% plot(y(1001:100000,1),y(1001:100000,2),'.');
% title('driver')
% subplot(1,2,2)
% plot(y(1001:100000,4),y(1001:100000,5),'.');
% title('response with C=1')
