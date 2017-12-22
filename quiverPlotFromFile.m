function quiverPlotFromFile( filenameField,filenamePotential,siz)

fileIDField = fopen(filenameField,'r');
fileIDPotential = fopen(filenamePotential,'r');
format = '%f %f';

data = fscanf(fileIDField,format,[2 Inf]);

data = data';

U = data(:,1);
V = data(:,2);
newU=reshape(U,siz,siz);
newV=reshape(V,siz,siz);
size(newU)
size(newV)

pot = fscanf(fileIDPotential,'%f');

pot = reshape(pot,siz,siz)';

size(pot)
[X,Y] = meshgrid(1:siz,1:siz);
size(X)
size(Y)
figure;
surf(X,Y,pot);
figure;
quiver(newV,newU);
fclose(fileIDField);
fclose(fileIDPotential);
end

