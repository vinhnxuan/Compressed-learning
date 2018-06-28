path(path,genpath(pwd));
nx=28;
ny=28;


load('matrix_1/W_matrix_0_196_0')

w1=w;


load('matrix_1/W_matrix_100_196_0')


w2=w;


hist1=coherence(w1.');
hist2=coherence(w2.');

nbins=0:0.001:1;

figure(5)
plot(nbins,hist1)


figure(6)
plot(nbins,hist2)


phi1=w1.';
phi2=w2.';

images = loadMNISTImages('t10k-images.idx3-ubyte');


num=10000;


snr=20:10:80;

null_A_1= null(phi1);
null_A_2= null(phi2);

psnr_arr1=zeros(length(snr),1);
psnr_arr2=zeros(length(snr),1);

mmse_arr1=zeros(length(snr),1);
mmse_arr2=zeros(length(snr),1);

count=1;
for sn=20:10:80
    
    psn1=0;
    psn2=0;

    mse1=0;
    mse2=0;
    ssim1=0;
    ssim2=0;
    for i =1:num
        disp(sprintf('Iteration %d',i))
        image=images(:,i);
              
        [img_sort,idx]=sort(abs(image),'descend');
%         
        KK=randi([15,35],1,1);
% 
        image(idx(KK:end))=0;
        
        B0 = reshape (image, [nx ny]);
        

        y1=phi1*image;
        y2=phi2*image;

        y1 = awgn(y1,sn,'measured');
        y2 = awgn(y2,sn,'measured');

        x_particular= pinv(phi1.'*phi1)*phi1.'*y1;
        
        x_particular2= pinv(phi2.'*phi2)*phi2.'*y2;
        
        zero_tolerance=1e-12;
%         
        U=cs_null_adagrad_new(y1,phi1,null_A_1,x_particular,zero_tolerance,'null_kf',0.98);
        U2=cs_null_adagrad_new(y2,phi2,null_A_2,x_particular2,zero_tolerance,'null_kf',0.98);
        
        
        
        
        U = reshape (U, [nx ny]);
        U2 = reshape (U2, [nx ny]);




        peakval=max(B0(:));

        [peaksnr]=psnr(double(U),B0,peakval);
        [peaksnr2]=psnr(double(U2),B0,peakval);

        mse1=mse1+immse(double(U),B0);
        mse2=mse2+immse(double(U2),B0);

        ssim1=ssim1+ssim(double(U),B0);
        ssim2=ssim2+ssim(double(U2),B0);

        psn1=psn1+peaksnr;
        psn2=psn2+peaksnr2;
    %     figure(123); 
    %     imshow(B0,[]);
        psn1_cur=psn1/i
        psn2_cur=psn2/i

        mse1_cur=mse1/i
        mse2_cur=mse2/i
    end
    mmse_arr1(count)=mse1/num;
    mmse_arr2(count)=mse2/num;
    
    psnr_arr1(count)=psn1/num;
    psnr_arr2(count)=psn2/num;
    count=count+1;
    
    figure(333)
    plot(snr,mmse_arr1,'b')
    hold on
    plot(snr,mmse_arr2,'r')
    hold off
    drawnow
    
    figure(444)
    plot(snr,psnr_arr1,'b')
    hold on
    plot(snr,psnr_arr2,'r')
    hold off
    drawnow
    
end








