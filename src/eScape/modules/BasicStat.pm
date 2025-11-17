package BasicStat;
use strict;

sub sum {
 	my $aref = $_[0];
	my @data = @{$aref};
	my $sum;
        foreach (@data) {
                $sum = $sum + $_;
        }
	
	return $sum;
}

sub median {
	my $aref = $_[0];
	my @data = @{$aref};
	my $Ndata = $#data + 1;
	my $mod = $Ndata%2;

	my $median;
	if ($mod == 0) {
		#Average two values in the middle.
		my $i1 = $Ndata/2;
		my $i2 = $i1 + 1;
		my @data = ($data[$i1], $data[$i2]);
		$median = mean(\@data);
	}
	else {
		my $i = ( (($Ndata-1)/2) + 1 );
		$median = $data[$i];
	}
	return $median;
}

sub mean {
	my $aref = $_[0];
	my @data = @{$aref};
	my $mean;

	my $sum = 0;
	my $N = 0;	
	foreach my $data (@data) {
	   if ($data =~ /\d+/) {
		$sum = $sum + $data;
		$N++;
	   }
	}

	$mean = $sum/$N;
	return $mean;
}

sub stdev {
	my ($mean, $aref) = @_;

	my @data = @{$aref};
	my $sd = 0;
	my $var = 0;
      
        my $N;
	foreach my $data (@data) {
	   if ($data =~ /\d/) {
		$var = $var + ($data - $mean)**2;
		$N++;
	   }
	}
	$sd = ($var/($N))**(1/2);
	return $sd;
}

sub accuracy {
	my $aref = $_[0];
	my ($tp, $tn, $fp, $fn) = @{$aref};
	# print "TP: $tp TN: $tn FP: $fp FN: $fn \n";
	my $accuracy;

	if (@{$aref} == (0,0,0,0)) {
		$accuracy = 'N/A';
	}
	else {
		$accuracy = ($tp + $tn) / ($tp + $fp + $tn + $fn);
	}

	return $accuracy;
}

sub precision {
        my $aref = $_[0];
        my ($tp, $tn, $fp, $fn) = @$aref;
	my $precision;
	
	if ($tp == 0 && $fp == 0) {
		$precision = 'N/A';
	}
	else {
		$precision = $tp /($tp+$fp);
	}

	return $precision;
}

sub recall {
        my $aref = $_[0];
        my ($tp, $tn, $fp, $fn) = @$aref;
	my $recall;

	if ($tp == 0 && $fn == 0) {
		$recall = 'N/A';
	}
	else {
		$recall = $tp/ ($tp + $fn);
	}

	return $recall;
}

sub entropy {
  my ($aref) = @_;
  my $H;
  my @tmp = @{$aref};
        for my $i (0 .. $#tmp) {
          # print "CALC ENTROPY $i $tmp[$i]  $#tmp\n";
          my $data = $tmp[$i];
          if ($data == 0) {$data = 0.000001;}

          $tmp[$i] = $data * log2($data);
        }
  $H = sum(\@tmp);
  $H = -$H;

  return $H;
}

sub eentropy {
  my ($aref) = @_;
  my $H;
  my @tmp = @{$aref};
        for my $i (0 .. $#tmp) {
          # print "CALC ENTROPY $i $tmp[$i]  $#tmp\n";
          my $data = $tmp[$i];
          if ($data == 0) {$data = 0.000001;}

          $tmp[$i] = $data * log($data);
        }
  $H = BasicStat::sum(\@tmp);
  $H = -$H;
  $H = exp($H);

  return $H;
}

sub log2 {
        my $n = shift;
        return log($n)/log(2);
}

sub RRMSE {
  
  # RRMSE
  # (sum((pred-target)^2)/ sum((target-mtarget)^2)^1/2)

  my ($Paref, $Daref) = @_;
  my @PRED = @{$Paref};
  my @DATA = @{$Daref};
  my ($RRMSE, $error, $norm);
 
  my $mDATA = mean(\@DATA);
 
  for my $i (0 .. $#PRED) {
	my $tmp1 = ($PRED[$i] - $DATA[$i])**2;
	my $tmp2 = ($DATA[$i] - $mDATA)**2;
 	
	# print "PRED $PRED[$i] TARGET $DATA[$i]  ERROR $tmp1 NORM $tmp2 \n"; 
	$error += $tmp1;
	$norm += $tmp2;
  }

  $RRMSE = ($error/$norm)**(1/2);

  return $RRMSE; 

}
1;
