function [hist1]= coherence (Q_opt)


[m1,m2]=size(Q_opt);
    
    
Q_norm= zeros(m1,m2);

for i = 1:m2
    Q_norm(:,i)=Q_opt(:,i)/norm(Q_opt(:,i));
end


K=eye(m2)-Q_norm.'*Q_norm;


mu=max(abs(K(:)))

nbins=0:0.001:1;

hist1=hist(abs(K(:)),nbins);





