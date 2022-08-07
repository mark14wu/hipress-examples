hipress: clean
	nsys profile --stats=true -t cuda python hipress_mxnet_profiling.py

torchddp: clean
	nsys profile --stats=true -t cuda python torchddp_profiling.py

clean:
	rm -rf report*