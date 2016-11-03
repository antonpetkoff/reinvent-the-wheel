import Prelude hiding (not)

data PRF = Z | S | I Int Int | C PRF [PRF] | R PRF PRF

eval :: PRF -> [Int] -> Int
eval _ [] = error "No arguments"
eval Z [x] = 0
eval S [x] = x + 1
eval (I n i) xs = xs !! i
eval (C f gs) xs = eval f (map (\g -> eval g xs) gs)
eval (R f _) (0:xs) = eval f xs
eval (R f g) (y:xs) = eval g ((eval (R f g) (y-1:xs)):y-1:xs)
eval _ _ = error "Incorrect call"

not :: PRF
not = Z

plus :: PRF
plus = R (I 1 0) (C S [I 3 0])

mult :: PRF
mult = R Z (C plus [(I 3 0), (I 3 2)])

main = do
    print $ eval plus [10, 15]
    print $ eval mult [2, 5]
