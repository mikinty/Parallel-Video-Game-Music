N = [
    1,
    4,
    16,
    64,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    98304,
    131072
];

z = zeros(1, 15);
z(:) = 10;

parallelTime = [
8.51188755,
8.617859602,
8.658136845,
8.724551678,
8.728380203,
9.012953758,
9.260383368,
9.587109804,
10.21303654,
11.75171542,
14.7658546,
20.24373436,
31.25442457,
44.38340068,
154.7996364
];

seqTime = [
    33.3893373,
33.39363289,
33.42382145,
33.53398538,
33.96940446,
34.58017826,
35.773175,
38.25236225,
43.09434056,
51.77858377,
71.78258991,
106.7150185,
191.6852055,
242.8648696,
333.0364153
];

%%% Plotting %%%
semilogx(N, parallelTime, '--.', 'MarkerSize', 15) 
hold on
semilogx(N, seqTime, '--.', 'MarkerSize', 15)
title('Time of Parallel vs. Sequential Music Generation')
ylabel('Time (s)')
xlabel('Number of measures (log)')

figure;
semilogx(N, seqTime ./ parallelTime, '--.', 'MarkerSize', 15) 
hold on
semilogx(N, z, 'LineWidth', 3)
title('Speedup of Parallel vs. Sequential Music Generation')
ylabel('Speedup')
ylim([0, 12])
xlabel('Number of measures (log)')