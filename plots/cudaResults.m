N = [
12500000,
25000000,
125000000,
250000000
];

seq = [
    178352,
472437,
2200164,
4045996
];

par = [
45579,
110788,
540189,
1067036
];

%%% Plotting %%%
plot(N, par/1000, '--.', 'MarkerSize', 15) 
hold on
plot(N, seq/1000, '--.', 'MarkerSize', 15)
title('Time of Parallel vs. Sequential Music Generation')
ylabel('Time (ms)')
xlabel('Number of measures (log)')

figure;
plot(N, seq ./ par, '--.', 'MarkerSize', 15)
title('Speedup of Parallel vs. Sequential Music Generation')
ylabel('Speedup')
ylim([0, 11])
xlabel('Number of notes (log)')