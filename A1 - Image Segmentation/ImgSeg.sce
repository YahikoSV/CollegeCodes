J = im2double(imread('D:\College\Physics 186\AP 186 Act - 7\BBIGB.jpg'));


//%{
//  ---Short Tip ---
//I = imread('cgc.jpg');
//[count, cells] = imhist(I, 256);
//plot (cells,count);
//BW = I < 125;
//imshow(BW);
//%}

//%Extract Pixels (interactive)
R_J = J(:,:,1);
G_J = J(:,:,2);
B_J = J(:,:,3);


//%RGB to normalized chromnacity coordinates
Int_J = R_J+G_J+B_J;
Int_J(find(Int_J==0))=100000;
r_J = R_J./Int_J;
g_J = G_J./Int_J;

//%Cropped Img
//I = imcrop(J);
//I=imcrop(J,[250,110,50,70]) //skin
I=imcrop(J,[240,220,25,35]) 
imshow(I)
R_I = I(:,:,1);
G_I = I(:,:,2);
B_I = I(:,:,3);

Int_I = R_I+G_I+B_I;
Int_I(find(Int_I==0))=100000;
R_I = I(:,:,1);
G_I = I(:,:,2);
r_I = R_I./Int_I;
g_I = G_I./Int_I;



//%Histogram Backprojection

BINS = 32;
rint = round(r_I*(BINS-1)+1);
gint = round(g_I*(BINS-1)+1);
colors = gint(:) + (rint(:)-1)*BINS;
hist = zeros(BINS,BINS);

for row = 1:BINS
    for col = 1:(BINS-row+1)
        hist(row,col) = length (find(colors==(((col+(row-1)*BINS)))));
        
    end;
end;

J_rint = round(r_J*(BINS-1)+1);
J_gint = round(g_J*(BINS-1)+1);

ssize = size(J_rint);
Result = zeros(ssize(1),ssize(2));



figure;
(Matplot(hist*255/max(hist))); 
hist2 = round(hist*255/max(hist));

for row1 = 1:ssize(1)
    for col1 = 1:ssize(2)
        Result(row1,col1) = hist2(J_rint(row1,col1),J_gint(row1,col1));
        
    end;
end;

figure;
Matplot(Result)

//% Parametric Segmentation
//testr = r_J-mean2(r_I);
//p_r =  ( 1/(std2(r_I)*sqrt(2*pi)))*exp(-testr.^2/(2*std2(r_I)^2));
//
//testg = g_J-mean2(g_I);
//p_g =  ( 1/(std2(g_I)*sqrt(2*pi)))*exp(-testg.^2/(2*std2(g_I)^2));
//
//p_rg = p_r.*p_g;
//figure;
//imagesc(p_rg);
//colormap Gray;
//
//imwrite(Result,'D:\College\Physics 186\AP 186 Act - 7\greenball.csv')
//
csvWrite(Result,'D:\College\Physics 186\AP 186 Act - 7\skin.csv')
