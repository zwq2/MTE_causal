
function ret = Lorenz(t,x,C)
%     y = zeros(1,3);
%     y(1) = x(4);
%     y(2) = x(5);
%     y(3) = x(6);
%     C = varagin{1};
    
    dx1_dt = -6*(x(2)+x(3));
    dx2_dt = 6*(x(1)+0.2*x(2));
    dx3_dt = 6*(0.2+x(3)*(x(1)-5.7));
    dy1_dt = 10*(-x(4)+x(5));
    dy2_dt = 28*x(4) - x(5) - x(4)*x(6) + C*x(2)*x(2);
    dy3_dt = x(4)*x(5) - 8/3*x(6);
    
    ret = [dx1_dt;dx2_dt;dx3_dt;dy1_dt;dy2_dt;dy3_dt];
    
end
