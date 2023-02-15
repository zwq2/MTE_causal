function threshold = p_test(x,y,dim,tau,u,alpha,i,j,m,method)
% k表示置换的次数
k = 200;
MTEx2y = zeros(1,k);
for kk = 1:k
    xx = x(i:m+i);
    yy = y(j:m+j);
    if method=='kte'
        MTEx2y(kk) = kRTE2(xx,yy,dim,tau,u,alpha);
    end
    if method == 'dif'
        MTEx2y(kk) = KMTE_HD(xx,yy,dim,tau,u,alpha) - KMTE_HD(yy,xx,dim,tau,u,alpha)
        
    end
    if method == 'hig'
        MTEx2y(kk) = KMTE_HD(xx,yy,dim,tau,u,alpha);
    end
    
%     MTEx2y(kk) = kRTE2(xx,yy,dim,tau,u,alpha);

    
    i = i+ 1;
    j = j + 1;
end
Mean = mean(MTEx2y);
STD = std(MTEx2y);
threshold = Mean+3*STD;
end