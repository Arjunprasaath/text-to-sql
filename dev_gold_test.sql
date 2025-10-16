SELECT count(*) FROM singer	concert_singer
SELECT count(*) FROM singer	concert_singer
SELECT name ,  country ,  age FROM singer ORDER BY age DESC	concert_singer
SELECT name ,  country ,  age FROM singer ORDER BY age DESC	concert_singer
SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'	concert_singer
SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'	concert_singer
SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1	concert_singer
SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1	concert_singer
SELECT DISTINCT country FROM singer WHERE age  >  20	concert_singer
SELECT DISTINCT country FROM singer WHERE age  >  20	concert_singer
SELECT country ,  count(*) FROM singer GROUP BY country	concert_singer
SELECT country ,  count(*) FROM singer GROUP BY country	concert_singer
SELECT song_name FROM singer WHERE age  >  (SELECT avg(age) FROM singer)	concert_singer
SELECT song_name FROM singer WHERE age  >  (SELECT avg(age) FROM singer)	concert_singer
SELECT LOCATION ,  name FROM stadium WHERE capacity BETWEEN 5000 AND 10000	concert_singer
SELECT LOCATION ,  name FROM stadium WHERE capacity BETWEEN 5000 AND 10000	concert_singer
select max(capacity), average from stadium	concert_singer
select avg(capacity) ,  max(capacity) from stadium	concert_singer
SELECT name ,  capacity FROM stadium ORDER BY average DESC LIMIT 1	concert_singer
SELECT name ,  capacity FROM stadium ORDER BY average DESC LIMIT 1	concert_singer
SELECT count(*) FROM concert WHERE YEAR  =  2014 OR YEAR  =  2015	concert_singer
SELECT count(*) FROM concert WHERE YEAR  =  2014 OR YEAR  =  2015	concert_singer
SELECT T2.name ,  count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id GROUP BY T1.stadium_id	concert_singer
SELECT T2.name ,  count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id GROUP BY T1.stadium_id	concert_singer
SELECT T2.name ,  T2.capacity FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.year  >=  2014 GROUP BY T2.stadium_id ORDER BY count(*) DESC LIMIT 1	concert_singer
select t2.name ,  t2.capacity from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  >  2013 group by t2.stadium_id order by count(*) desc limit 1	concert_singer
SELECT YEAR FROM concert GROUP BY YEAR ORDER BY count(*) DESC LIMIT 1	concert_singer
SELECT YEAR FROM concert GROUP BY YEAR ORDER BY count(*) DESC LIMIT 1	concert_singer
SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)	concert_singer
SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)	concert_singer
SELECT country FROM singer WHERE age  >  40 INTERSECT SELECT country FROM singer WHERE age  <  30	concert_singer
SELECT name FROM stadium EXCEPT SELECT T2.name FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.year  =  2014	concert_singer
SELECT name FROM stadium EXCEPT SELECT T2.name FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.year  =  2014	concert_singer
SELECT T2.concert_name ,  T2.theme ,  count(*) FROM singer_in_concert AS T1 JOIN concert AS T2 ON T1.concert_id  =  T2.concert_id GROUP BY T2.concert_id	concert_singer
select t2.concert_name ,  t2.theme ,  count(*) from singer_in_concert as t1 join concert as t2 on t1.concert_id  =  t2.concert_id group by t2.concert_id	concert_singer
SELECT T2.name ,  count(*) FROM singer_in_concert AS T1 JOIN singer AS T2 ON T1.singer_id  =  T2.singer_id GROUP BY T2.singer_id	concert_singer
