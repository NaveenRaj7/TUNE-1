clc
clear all
close all

% Sample data path, where the unisens.xml file is located
nu = 1:60;
nu = nu';
t = 1;
all = [];
label = [];
% diff_all = [];
x = linspace(0,60, 60);
for p = 1:15
    
    o = num2str(p);
    path = strcat('C:\Users\rajna\Documents\MATLAB\TUNE1\renamed\New folder', o);
    jUnisensFactory = org.unisens.UnisensFactoryBuilder.createFactory();
    jUnisens = jUnisensFactory.createUnisens(path);
    % Read a binary file
    if(p==1)
        accEntry = jUnisens.getEntry('eda.csv');
    else
        accEntry = jUnisens.getEntry('eda.bin');
    end
    
    accData = accEntry.readScaled(accEntry.getCount()); % In accData will be the values of acc.bin
    jUnisens.closeAll();
    
    accData = accData(7:end);
    s = length(accData);
    ind = 1:31:1830;
    
    m1 = accData(1:round(s/4));
    m2 = accData(round(s/4)+1:round(2*s/4));
    m3 = accData(round(2*s/4)+1:round(3*s/4));
    m4 = accData(round(3*s/4)+1:round(4*s/4));
    
    e = m1(ind);
    d = m4(ind);
    
    m1_n = (m1 - min(m1)) ./ (max(m1) - min(m1));
    m4_n = (m4 - min(m4)) ./ (max(m4) - min(m4));
    
    easy = m1_n(ind);
    diff = m4_n(ind);
    
    easy = [nu, easy];
    easy = [t*ones(length(easy),1), easy];
    all = [all;easy];
    label = [label;0];
    t = t+1;
    diff = [nu, diff];
    diff = [t*ones(length(diff),1), diff];
    all = [all;diff];
    label = [label;1];
    t = t+1;
    
    
    for q = 1:5
        
        noiseSigma = q * 0.04 * e;
        noise = noiseSigma .* randn(1, length(e))';
        easy = e + noise;
%         plot(x, easy, 'r.', 'MarkerSize', 10);
%         hold on;
%         plot(x, e, 'b-', 'LineWidth', 3);
%         grid on;
%         hold off
%         
%         figure
        
        noiseSigma = q * 0.04 * d;
        noise = noiseSigma .* randn(1, length(d))';
        diff = d + noise;
%         plot(x, diff, 'r.', 'MarkerSize', 10);
%         hold on;
%         plot(x, d, 'b-', 'LineWidth', 3);
%         grid on;
%         hold off
        
        %normalization
        easy = (easy - min(easy))./(max(easy)-min(easy));
        diff = (diff - min(diff))./(max(diff)-min(diff));
        
        easy = [nu, easy];
        easy = [t*ones(length(easy),1), easy];
        all = [all;easy];
        label = [label;0];
        t = t+1;
        
        diff = [nu, diff];
        diff = [t*ones(length(diff),1), diff];
        all = [all;diff];
        label = [label;1];
        t = t+1;

             
%         n = num2str(q);
%         poly = strcat('poly', n);
%         
%         f_e = fit([1:length(e)]', e, poly);
%         f_d = fit([1:length(d)]', d, poly);
%         
%         
%         n_e = f_e(x);
%         n_d = f_d(x);
%         
%         easy = n_e + 0.5 * 2*(rand(1, length(n_e))'-.5);
%         %normalization
%         easy = (easy - min(easy))./(max(easy)-min(easy));
%         
%         diff = n_d + 0.5 * 2*(rand(1, length(n_d))'-.5);
%         %normalization
%         diff = (diff - min(diff))./(max(diff)-min(diff));
%         
%         %noisy = accData + 0.5 * 2*(rand(1, length(accData))'-.5);
%         %         m1 = noisy(1:round(s/4));
%         %         m2 = noisy(round(s/4)+1:round(2*s/4));
%         %         m3 = noisy(round(2*s/4)+1:round(3*s/4));
%         %         m4 = noisy(round(3*s/4)+1:round(4*s/4));
%         
%         %         m1 = (m1 - min(m1)) ./ (max(m1) - min(m1));
%         %         m4 = (m4 - min(m4)) ./ (max(m4) - min(m4));
%         %
%         %         easy = m1(ind);
%         %         diff = m4(ind);
%         
%         easy = smooth(easy);
%         diff = smooth(diff);
%         easy = [nu, easy];
%         easy = [t*ones(length(easy),1), easy];
%         easy_all = [easy_all;easy];
%         
%         diff = [nu, diff];
%         diff = [(t+90)*ones(length(diff),1), diff];
%         diff_all = [diff_all;diff];
%         
%         t = t+1;
    end
    
    %save_path = strcat('C:\Users\rajna\Documents\MATLAB\TUNE1\renamed\pro\set_', o);
    %save(save_path, 'easy', 'diff');
end
