load('../dataset/TrainingSamplesDCT_8.mat');
zigzag = load('../dataset/Zig-Zag Pattern.txt');
cheetah = imread('../dataset/cheetah.bmp');
cheetah_mask = imread('../dataset/cheetah_mask.bmp');
target = im2double(cheetah);
mask = im2double(cheetah_mask);

training_BG = TrainsampleDCT_BG;
training_FG = TrainsampleDCT_FG;

[row_BG, col_BG] = size(training_BG);
[row_FG, col_FG] = size(training_FG);
[row_TG, col_TG] = size(target);

zigzag = zigzag + 1;

epoch = 100;

prior_BG = row_BG / (row_BG + row_FG);
prior_FG = row_FG / (row_BG + row_FG);

% pick cheetah if (p(x | grass) / p(x | cheetah)) < threshold
threshold = prior_FG / prior_BG;

mean_FG = zeros(1, 64);
mean_BG = zeros(1, 64);

cov_FG = zeros(64, 64);
cov_BG = zeros(64, 64);

for r = 1:row_FG
    cov_FG = cov_FG + training_FG(r,:)' * training_FG(r,:);
    mean_FG = mean_FG + training_FG(r,:);
end

for r = 1:row_BG
    cov_BG = cov_BG + training_BG(r,:)' * training_BG(r,:);
    mean_BG = mean_BG + training_BG(r,:);
end

mean_FG = mean_FG / row_FG;
mean_BG = mean_BG / row_BG;

cov_FG = (cov_FG / row_FG) - mean_FG' * mean_FG;
cov_BG = (cov_BG / row_BG) - mean_BG' * mean_BG;

C = 8;

pi_FG = rand(1, C);
pi_FG = pi_FG / sum(pi_FG);

pi_BG = rand(1, C) + 1;
pi_BG = pi_BG / sum(pi_BG);

mu_FG = rand(64, C);
mu_BG = rand(64, C);
for c=1:C
    mu_FG(:, c) = mu_FG(:, c) + mean_FG';
    mu_BG(:, c) = mu_BG(:, c) + mean_BG';
end

sigma_FG = 1 + rand(64, C);
sigma_BG = 1 + rand(64, C);

for itr=1:epoch
    h_BG = zeros(row_BG, C);
    for i=1:row_BG
        for j=1:C
            h_BG(i, j) = mvn(training_BG(i, :), mu_BG(:, j)', ...
                diag(sigma_BG(:, j)')) * pi_BG(j);
        end
        h_BG(i, :) = h_BG(i, :) ./ sum(h_BG(i, :));
    end

    for j=1:C
        pi_BG(j) = sum(h_BG(:, j)) / row_BG;
        temp_mu = zeros(64, 1);
        for i=1:row_BG
            temp_mu = temp_mu + h_BG(i, j) * training_BG(i, :)';
        end
        
        mu_BG(:, j) = temp_mu / sum(h_BG(:, j));
        temp_sigma = zeros(64, 1);
        for i=1:row_BG
            temp_sigma = temp_sigma + h_BG(i, j) * ((training_BG(i, :)' - mu_BG(:, j)).^2);
        end
        sigma_BG(:, j) = temp_sigma / sum(h_BG(:, j));
        sigma_BG(sigma_BG < 0.0001) = 0.0001;
    end
end

for itr=1:epoch
    h_FG = zeros(row_FG, C);
    for i=1:row_FG
        for j=1:C
            h_FG(i, j) = mvn(training_FG(i, :), mu_FG(:, j)', ...
                diag(sigma_FG(:, j)')) * pi_FG(j);
        end
        h_FG(i, :) = h_FG(i, :) ./ sum(h_FG(i, :));
    end

    for j=1:C
        pi_FG(j) = sum(h_FG(:, j)) / row_FG;
        temp_mu = zeros(64, 1);
        for i=1:row_FG
            temp_mu = temp_mu + h_FG(i, j) * training_FG(i, :)';
        end
        
        mu_FG(:, j) = temp_mu / sum(h_FG(:, j));
        temp_sigma = zeros(64, 1);
        for i=1:row_FG
            temp_sigma = temp_sigma + h_FG(i, j) * ((training_FG(i, :)' - mu_FG(:, j)).^2);
        end
        sigma_FG(:, j) = temp_sigma / sum(h_FG(:, j));
        sigma_FG(sigma_FG < 0.0001) = 0.0001;
    end
end

sq_sigma_FG = zeros(64, 64, C);
sq_sigma_BG = zeros(64, 64, C);
for i = 1:C
    sq_sigma_FG(:, :, i) = diag(sigma_FG(:, i));
    sq_sigma_BG(:, :, i) = diag(sigma_BG(:, i));
end

gm_FG = gmdistribution(mu_FG', sq_sigma_FG, pi_FG);
gm_BG = gmdistribution(mu_BG', sq_sigma_BG, pi_BG);

A = zeros(row_TG, col_TG);

for r = 5:row_TG-3
    for c = 5:col_TG-3
        block = target(r - 4:r + 3, c - 4:c + 3);
        dctBlock = dct2(block);
        X = zeros(1, 64);
        for i = 1:8
            for j = 1:8
                X(zigzag(i, j)) = dctBlock(i, j);
            end
        end
        A(r, c) = int8(pdf(gm_BG, X)/ ...
            pdf(gm_FG, X) <= threshold);
    end
end

figure;

subplot(1, 3, 1);
imagesc(target);
axis off
colormap(gray(255));
axis equal tight;

subplot(1, 3, 2);
imagesc(mask);
axis off
colormap(gray(255));
axis equal tight;

subplot(1, 3, 3);
imagesc(A);
axis off
colormap(gray(255));
axis equal tight;

error = 0;
for r = 1:row_TG
    for c = 1:col_TG
        if (A(r, c) ~= mask(r, c))
            error = error + 1;
        end
    end
end

error_rate = error / (row_TG * col_TG);
disp(error_rate);