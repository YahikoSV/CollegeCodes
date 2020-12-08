A = zeros(53,20);

A(4:8,7:11)=1; "5by5square"

A(14:23,4:13)=1;
A(18:19,8:9)=0; "10by10 with 2x2 holllow"

A(30:40,9)=1;
A(35,4:14)=1; "cross"

A(45,10)=1; "triangle"
A(46,8:10)=1;
A(47,7:10)=1;


SE1 = CreateStructureElement('custom',[ %f %t %f; %t %t %t; %f %t %f]); "cross, 3 pixels long, one pixel thick."
SE2 = CreateStructureElement('custom',[ %t %t ; %t %t]);  "2x2"
SE3 = CreateStructureElement('custom',[ %t %t]);  "2x1"
SE4 = CreateStructureElement('custom',[ %t ; %t]); "1x2"
SE5 = CreateStructureElement('custom',[ %f %t ; %t %f]); ' A diagonal line"
SE6 = CreateStructureElement('custom',[ %t %f ; %f %t]); ' Another diagonal line"


A1 = DilateImage(A,SE1);
A2 = DilateImage(A,SE2);
A3 = DilateImage(A,SE3);
A4 = DilateImage(A,SE4);
A5 = DilateImage(A,SE5);
A6 = DilateImage(A,SE6);
imshow(A)

B1 = ErodeImage(A,SE1);
B2 = ErodeImage(A,SE2);
B3 = ErodeImage(A,SE3);
B4 = ErodeImage(A,SE4);
B5 = ErodeImage(A,SE5);
B6 = ErodeImage(A,SE6);'

imwrite(A,'D:\College\Physics 186\Activity 8\Orig.png')

C = zeroes(25,5)

C(2:3,2:3)=1;

C(7,2:3)=1;

C(12:13,2)=1;

C(17:19,3)=1;
C(18,2:4)=1;

C(22,2)=1;
C(23,3)=1;