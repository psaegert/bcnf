# Loop through all files in ./configs/runs/hybrid and run bcnf train -c {{BCNF_ROOT}}/configs/runs/hybrid/<filename>

# Loop through all files in ./configs/runs/hybrid
for file in ./configs/runs/hybrid/*; do
    # Run bcnf train -c {{BCNF_ROOT}}/configs/runs/hybrid/<filename>
    bcnf train -c {{BCNF_ROOT}}/configs/runs/hybrid/$(basename $file)
done