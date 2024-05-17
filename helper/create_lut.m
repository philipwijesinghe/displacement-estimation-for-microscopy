


[x, y] = meshgrid(1:1000);


figure;
imagesc(x);
colormap(gray(256))


%%
cmap = dicm_c3;
cmap = uint8(cmap * 255);
fname = 'dcim_c3.lut';
fid = fopen(fname, 'w+');
fprintf(fid, "Index\tRed\tGreen\tBlue\n");
for idx = 1:256
    fprintf(fid, "%i\t%i\t%i\t%i\n", idx - 1, cmap(idx, 1), cmap(idx, 2), cmap(idx, 3));
end
fclose(fid);