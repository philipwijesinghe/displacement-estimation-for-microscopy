// Start with a video stack in imagej
model="deformnet"  // SELECT MODEL NAME AS INSTALLED IN DEEPIMAGEJ <!>
name=getTitle();
getDimensions(w, h, channels, slices, frames);

for (i = 1; i < slices; i++) {
	selectImage(name);
	i2 = i + 1;

	// Create network input
	run("Make Substack...", "slices="+i+","+i+","+i+","+i2+","+i2+","+i2);
	rename("img_pair_"+i);

	// normalise to 0,1 (Assumes 8-bit Grayscale; EDIT THIS SECTION FOR YOUR DATA TYPE) <!>
	run("32-bit");
	run("Divide...", "value=256 stack");
	setMinAndMax(0, 1);
	run("Properties...", "channels=6 slices=1 frames=1 pixel_width=1.0000 pixel_height=1.0000 voxel_depth=1.0000");

	// Run model
	run("DeepImageJ Run", "model="+model+" format=Pytorch preprocessing=[no preprocessing] postprocessing=[no postprocessing] axes=C,Y,X tile=6,1024,1024 logging=Normal");

	// Format output and append to stacks 
	rename("output_"+i);
	run("Split Channels");
	
	selectImage("C1-output_"+i);
	rename("ux_output_"+i);
	run("Grays");
	if (i==1) {
		rename("ux_output");
	} else {
		run("Concatenate...", "open image1=ux_output image2=ux_output_"+i+" image3=[-- None --]");
		rename("ux_output");
	}

	selectImage("C2-output_"+i);
	rename("uy_output_"+i);
	run("Grays");
	if (i==1) {
		rename("uy_output");
	} else {
		run("Concatenate...", "open image1=uy_output image2=uy_output_"+i+" image3=[-- None --]");
		rename("uy_output");
	}

	// Cleanup
	selectImage("img_pair_"+i);
	close();
}

