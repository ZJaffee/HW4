package edu.cwru.sepia.agent;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionFeedback;
import edu.cwru.sepia.action.ActionResult;
import edu.cwru.sepia.action.TargetedAction;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.DeathLog;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.history.History.HistoryView;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.State.StateView;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;

import java.io.*;
import java.util.*;
import java.util.Map.Entry;

public class RLAgent extends Agent {

    /**
     * Set in the constructor. Defines how many learning episodes your agent should run for.
     * When starting an episode. If the count is greater than this value print a message
     * and call sys.exit(0)
     */
    public final int numEpisodes;

    /**
     * List of your footmen and your enemies footmen
     */
    private Set<Integer> myFootmen;
    private Set<Integer> enemyFootmen;

    /**
     * Convenience variable specifying enemy agent number. Use this whenever referring
     * to the enemy agent. We will make sure it is set to the proper number when testing your code.
     */
    public static final int ENEMY_PLAYERNUM = 1;

    /**
     * Set this to whatever size your feature vector is.
     */
    public static final int NUM_FEATURES = 3;

    /** Use this random number generator for your epsilon exploration. When you submit we will
     * change this seed so make sure that your agent works for more than the default seed.
     */
    public final Random random = new Random(12345678);

    /**
     * Your Q-function weights.
     */
    public Double[] weights;

    /**
     * These variables are set for you according to the assignment definition. You can change them,
     * but it is not recommended. If you do change them please let us know and explain your reasoning for
     * changing them.
     */
    public final double gamma = 0.9;
    public final double learningRate = .0001;
    public /*final*/ double epsilon = 0.02;
    
    private List<Double> rewardsFromCurrentEpisode;
    private List<Double> averageRewards;
    private List<Double> averageRewardsOverFiveEpisodes;
    private boolean lastExploit = false;
    
    public Map <Integer, Double[] > previousFeatures;
    public Map<Integer, Double> cumulativeReward;
    public Map<Integer, Set<Integer>> beingAttackedBy;
    
    public int episodeNum = 0;
    public boolean explorationEpisode;
    
    public Map<Integer, Integer> lastAssignedAMove;
    
    private enum attackedByStatus {NONE, ATTACKED_BY_TARGET, NOT_ATTACKED_BY_TARGET};
    private Map<Integer, attackedByStatus> attackedBy;
    private int winCount = 0;
    

    public RLAgent(int playernum, String[] args) {
        super(playernum);

        if (args.length >= 1) {
            numEpisodes = Integer.parseInt(args[0]);
            System.out.println("Running " + numEpisodes + " episodes.");
        } else {
            numEpisodes = 10;
            System.out.println("Warning! Number of episodes not specified. Defaulting to 10 episodes.");
        }

        boolean loadWeights = false;
        if (args.length >= 2) {
            loadWeights = Boolean.parseBoolean(args[1]);
        } else {
            System.out.println("Warning! Load weights argument not specified. Defaulting to not loading.");
        }

        if (loadWeights) {
            weights = loadWeights();
        } else {
            // initialize weights to random values between -1 and 1
            weights = new Double[NUM_FEATURES];
            weights[0] = 1.0;
            for (int i = 1; i < weights.length; i++) {
                weights[i] = random.nextDouble() * 2 - 1;
            }
        }
        
        cumulativeReward = new HashMap<Integer, Double>();
        previousFeatures = new HashMap<Integer, Double[]>();
        beingAttackedBy = new HashMap<Integer, Set<Integer>>();
        lastAssignedAMove = new HashMap<Integer, Integer>();
        attackedBy = new HashMap<Integer, attackedByStatus>();
        averageRewards = new ArrayList<Double>();
        averageRewardsOverFiveEpisodes = new ArrayList<Double>();
    }

    /**
     * We've implemented some setup code for your convenience. Change what you need to.
     */
    @Override
    public Map<Integer, Action> initialStep(State.StateView stateView, History.HistoryView historyView) {

        //if( ((episodeNum++) / 5 ) % 2 == 0){
    	if( ((episodeNum) / 5) % 3 != 0){
        	//System.out.println("EXPLORING in this episode.");
        	explorationEpisode = true;
        }else{
        	//System.out.println("EXPLOITING in this episode.");
        	explorationEpisode = false;
        	lastExploit = episodeNum % 15 == 4;
        }
    	episodeNum++;

        // Find all of your units
        myFootmen = new HashSet<>();
        for (Integer unitId : stateView.getUnitIds(playernum)) {
            Unit.UnitView unit = stateView.getUnit(unitId);

            String unitName = unit.getTemplateView().getName().toLowerCase();
            if (unitName.equals("footman")) {
                myFootmen.add(unitId);
                previousFeatures.put(unitId, new Double[weights.length]);
            } else {
                System.err.println("Unknown unit type: " + unitName);
            }
        }

        // Find all of the enemy units
        enemyFootmen = new HashSet<>();
        for (Integer unitId : stateView.getUnitIds(ENEMY_PLAYERNUM)) {
            Unit.UnitView unit = stateView.getUnit(unitId);

            String unitName = unit.getTemplateView().getName().toLowerCase();
            if (unitName.equals("footman")) {
                enemyFootmen.add(unitId);
            } else {
                System.err.println("Unknown unit type: " + unitName);
            }
        }
        
        rewardsFromCurrentEpisode = new ArrayList<Double>();

        return middleStep(stateView, historyView);
    }

    /**
     * You will need to calculate the reward at each step and update your totals. You will also need to
     * check if an event has occurred. If it has then you will need to update your weights and select a new action.
     *
     * If you are using the footmen vectors you will also need to remove killed units. To do so use the historyView
     * to get a DeathLog. Each DeathLog tells you which player's unit died and the unit ID of the dead unit. To get
     * the deaths from the last turn do something similar to the following snippet. Please be aware that on the first
     * turn you should not call this as you will get nothing back.
     *
     * for(DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() -1)) {
     *     System.out.println("Player: " + deathLog.getController() + " unit: " + deathLog.getDeadUnitID());
     * }
     *
     * You should also check for completed actions using the history view. Obviously you never want a footman just
     * sitting around doing nothing (the enemy certainly isn't going to stop attacking). So at the minimum you will
     * have an even whenever one your footmen's targets is killed or an action fails. Actions may fail if the target
     * is surrounded or the unit cannot find a path to the unit. To get the action results from the previous turn
     * you can do something similar to the following. Please be aware that on the first turn you should not call this
     *
     * Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
     * for(ActionResult result : actionResults.values()) {
     *     System.out.println(result.toString());
     * }
     *
     * @return New actions to execute or nothing if an event has not occurred.
     */
    @Override
    public Map<Integer, Action> middleStep(State.StateView stateView, History.HistoryView historyView) {
    	Map<Integer, Action> ret = new HashMap<Integer, Action>();
        Set<Integer> inactiveUnits;
        if(stateView.getTurnNumber() != 0){
        	inactiveUnits = calcRewardsAndGetInactiveUnits(stateView, historyView,ret);
        	
        }else{
        	inactiveUnits = new HashSet<Integer>();
        	inactiveUnits.addAll(myFootmen);
        }
        for(Integer footmanId : inactiveUnits){
        	ret.put(footmanId, Action.createCompoundAttack(footmanId, selectAction(stateView, historyView, footmanId, false)));
        	cumulativeReward.put(footmanId, -0.1);
        	lastAssignedAMove.put(footmanId, 0);
        }
        //System.out.println(ret);
    	return ret;
    }

    /**
     * Here you will calculate the cumulative average rewards for your testing episodes. If you have just
     * finished a set of test episodes you will call out testEpisode.
     *
     * It is also a good idea to save your weights with the saveWeights function.
     */
    @Override
    public void terminalStep(State.StateView stateView, History.HistoryView historyView) {

    	removeDead(stateView, historyView);
    	updateWin();
        // MAKE SURE YOU CALL printTestData after you finish a test episode.
    	if(!explorationEpisode){
	    	double sum = 0.0;
	    	for(Double d : rewardsFromCurrentEpisode){
	    		sum += d;

	    		//System.out.println("Reward: "+d);
	    	}
	    	//System.out.println("Sum: "+sum+"  Size:"+rewardsFromCurrentEpisode.size());
	    	double avg = sum/rewardsFromCurrentEpisode.size();
	    	averageRewardsOverFiveEpisodes.add(avg);
	    	if(lastExploit){
	    		sum = 0.0;
	    		for(Double d : averageRewardsOverFiveEpisodes){
		    		sum += d;
		    	}
	    		avg = sum / averageRewardsOverFiveEpisodes.size();
	    		averageRewards.add(avg);
	    		printTestData(averageRewards);
	    		
	    		averageRewardsOverFiveEpisodes.clear();
	    		System.out.println("We won "+winCount+" out of 5.");
	    		/*if(winCount >= 4 && epsilon > 0.01){
	    			epsilon -= 0.01;
	    		}else{
	    			epsilon = 0.0;
	    		}*/
	    		winCount = 0;
	    		/*if(avg > 63){
	    			epsilon = 0.0;
	    		}*/
	    	}
	    	//rewardsFromCurrentEpisode.clear();
    	}else if(((episodeNum) / 5) % 3 == 0){
    		System.out.println("We won "+winCount+" out of 10.");
    		winCount = 0;
    	}
    	else if(episodeNum >= 25){
    		//epsilon = epsilon <= 0.3 ? 0 : epsilon - 0.03;
    	}
    	
    	/*if(explorationEpisode){
    		epsilon = epsilon >= 0.05 ? epsilon - 0.05 : 0.0;
    	}*/
    	
    	
    	
        // Save your weights
        saveWeights(weights);

    }

    private void updateWin() {
    	if(myFootmen.isEmpty() && !enemyFootmen.isEmpty()){
    		//System.out.println("The enemy won, with "+enemyFootmen.size()+" units left.");
    	}else{
    		//System.out.println("We won, with "+myFootmen.size()+" units left!");
    		winCount++;
    	}
	}

	private void removeDead(StateView stateView, HistoryView historyView) {
    	for(DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() - 1)) {
    		if(deathLog.getController() == playernum){
    			myFootmen.remove(deathLog.getDeadUnitID());
    		}else{
    			enemyFootmen.remove(deathLog.getDeadUnitID());
    		}
    	}
	}

	/**
     * Calculate the updated weights for this agent. 
     * @param oldWeights Weights prior to update
     * @param oldFeatures Features from (s,a)
     * @param totalReward Cumulative discounted reward for this footman.
     * @param stateView Current state of the game.
     * @param historyView History of the game up until this point
     * @param footmanId The footman we are updating the weights for
     * @return The updated weight vector.
     */
    public void updateWeights(State.StateView stateView, History.HistoryView historyView, int footmanId) {
        Double[] oldFeatures = previousFeatures.get(footmanId);
        double actualReward = cumulativeReward.get(footmanId);
        rewardsFromCurrentEpisode.add(actualReward);
        double qVal = dotProduct(weights, oldFeatures);
        
       // System.out.println("Old features: "+Arrays.toString(oldFeatures));
        //System.out.println("Reward: "+actualReward);
        
        double predictedQ = myFootmen.contains(footmanId) ? calcQValue(stateView, historyView, footmanId, selectAction(stateView, historyView, footmanId, true)) : 0;
        double LVal = -(actualReward - qVal + gamma*predictedQ);
       // System.out.println("Old weights: "+Arrays.toString(weights));
        
        for(int i = 1; i < weights.length; i++){
        	//not sure if this is right
        	weights[i] -= learningRate*LVal*oldFeatures[i];
        }
        //System.out.println("New weights: "+Arrays.toString(weights));
        if(!myFootmen.contains(footmanId)){
        	previousFeatures.remove(footmanId);
        	cumulativeReward.remove(footmanId);
        	for(Set<Integer> myUnits : beingAttackedBy.values()){
        		myUnits.remove(footmanId);
        	}
        }
    }

    /**
     * Given a footman and the current state and history of the game select the enemy that this unit should
     * attack. This is where you would do the epsilon-greedy action selection.
     *
     * @param stateView Current state of the game
     * @param historyView The entire history of this episode
     * @param attackerId The footman that will be attacking
     * @param selectBest 
     * @return The enemy footman ID this unit should attack
     */
    public int selectAction(State.StateView stateView, History.HistoryView historyView, int attackerId, boolean selectBest) {
    	
    	int enemyToAttack = -1;
    	//If it is not an exploration episode, or it is an exploration episode but we decided to choose the "best" option
    	if(selectBest || !explorationEpisode || random.nextDouble() >= epsilon){
    	
	        double bestQ = Double.NEGATIVE_INFINITY;
	    	for(Integer enemyId : enemyFootmen){
	        	double curQVal = calcQValue(stateView, historyView, attackerId, enemyId);
	        	if(curQVal > bestQ){
	        		bestQ = curQVal;
	        		enemyToAttack = enemyId;
	        	}
	        } 	
    	}
    	//If it is an exploration episode, and we decided not to go with the best option
    	else{
    		int enemyIndToAttack = random.nextInt(enemyFootmen.size());
    		int currentInd = 0;
    		for(Integer enemyId : enemyFootmen){
    			if(currentInd == enemyIndToAttack){
    				enemyToAttack = enemyId;
    				break;
    			}
    			currentInd++;
    		}
    	}
    	

    	if(!selectBest){
	    	for(Set<Integer> attackers : beingAttackedBy.values()){
	    		attackers.remove(attackerId);
	    	}
	    	
	    	previousFeatures.put(attackerId, calculateFeatureVector(stateView, historyView, attackerId, enemyToAttack));
	    	if(beingAttackedBy.containsKey(enemyToAttack)){
	    		beingAttackedBy.get(enemyToAttack).add(attackerId);
	    	}else{
	    		Set<Integer> attacking = new HashSet<Integer>();
	    		attacking.add(attackerId);
	    		beingAttackedBy.put(enemyToAttack, attacking);
	    	}
    	}
    	
    	return enemyToAttack;
    }

    /**
     * Given the current state and the footman in question calculate the reward received on the last turn.
     * This is where you will check for things like Did this footman take or give damage? Did this footman die
     * or kill its enemy. Did this footman start an action on the last turn? See the assignment description
     * for the full list of rewards.
     *
     * Remember that you will need to discount this reward based on the timestep it is received on. See
     * the assignment description for more details.
     *
     * As part of the reward you will need to calculate if any of the units have taken damage. You can use
     * the history view to get a list of damages dealt in the previous turn. Use something like the following.
     *
     * for(DamageLog damageLogs : historyView.getDamageLogs(lastTurnNumber)) {
     *     System.out.println("Defending player: " + damageLog.getDefenderController() + " defending unit: " + \
     *     damageLog.getDefenderID() + " attacking player: " + damageLog.getAttackerController() + \
     *     "attacking unit: " + damageLog.getAttackerID());
     * }
     *
     * You will do something similar for the deaths. See the middle step documentation for a snippet
     * showing how to use the deathLogs.
     *
     * To see if a command was issued you can check the commands issued log.
     *
     * Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(playernum, lastTurnNumber);
     * for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
     *     System.out.println("Unit " + commandEntry.getKey() + " was command to " + commandEntry.getValue().toString);
     * }
     *
     * @param stateView The current state of the game.
     * @param historyView History of the episode up until this turn.
     * @param actions 
     * @param footmanId The footman ID you are looking for the reward from.
     * @return The current reward
     */
    public Set<Integer> calcRewardsAndGetInactiveUnits(State.StateView stateView, History.HistoryView historyView, Map<Integer, Action> actions) {
    	int lastTurnNumber = stateView.getTurnNumber() -1;
    	Set<Integer> inactiveUnits = new HashSet<Integer>();
    	Set<Integer> myDead = new HashSet<Integer>();
    	Set<Integer> hadEvent = new HashSet<Integer>();
    	Set<Integer> deadEnemies = new HashSet<Integer>();
    	for(DeathLog deathLog : historyView.getDeathLogs(lastTurnNumber)) {
    		if(deathLog.getController() == playernum){
    			int unitId = deathLog.getDeadUnitID();
    			hadEvent.add(unitId);
    			myFootmen.remove(deathLog.getDeadUnitID());
    			myDead.add(deathLog.getDeadUnitID());
    			lastAssignedAMove.remove(unitId);
    			cumulativeReward.put(unitId, cumulativeReward.get(unitId) - 100);
    		}else{
    			int deadEnemy = deathLog.getDeadUnitID();
    			enemyFootmen.remove(deadEnemy);
    			for(Integer myUnit : beingAttackedBy.get(deadEnemy)){
    				//punishes footmen teaming up on others -- should make sure there is a feature to offset this
    				cumulativeReward.put(myUnit, cumulativeReward.get(myUnit) + (100.0/*/beingAttackedBy.get(deadEnemy).size()*/));
        			inactiveUnits.add(myUnit);
        			hadEvent.add(myUnit);
    			}
    			//beingAttackedBy.remove(deadEnemy);
    			deadEnemies.add(deadEnemy);
    		}
    	}
    	
    	
    	for(Integer myUnit : myFootmen){
    		attackedBy.put(myUnit, attackedByStatus.NONE);
    	}
    	for(DamageLog damageLog : historyView.getDamageLogs(lastTurnNumber)) {
    		if(damageLog.getDefenderController() == playernum){
    			int myUnit = damageLog.getDefenderID();
    			int enemyId = damageLog.getAttackerID();
    			cumulativeReward.put(myUnit, cumulativeReward.get(myUnit) - damageLog.getDamage());
    			if(beingAttackedBy.get(enemyId) != null && beingAttackedBy.get(enemyId).contains(myUnit)){
    				attackedBy.put(myUnit, attackedByStatus.ATTACKED_BY_TARGET);
    			}else if(attackedBy.get(myUnit) == attackedByStatus.NONE){
    				attackedBy.put(myUnit, attackedByStatus.NOT_ATTACKED_BY_TARGET);
    			}
    		}else{
    			int enemyUnit = damageLog.getDefenderID();;
    			for(Integer myUnit : beingAttackedBy.get(enemyUnit)){
    				//System.out.println(cumulativeReward.get(myUnit));
    				cumulativeReward.put(myUnit, cumulativeReward.get(myUnit) + (damageLog.getDamage()/*1.0/beingAttackedBy.get(enemyUnit).size()*/));
    			}
    		}
    	 }
    	
    	Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, lastTurnNumber);
    	for(Entry<Integer, ActionResult> result : actionResults.entrySet()) {
    		int myUnitId = result.getKey();
    		if(result.getValue().getFeedback() == ActionFeedback.COMPLETED || result.getValue().getFeedback() == ActionFeedback.FAILED){
    			//hadEvent.add(result.getKey());
    			if(myFootmen.contains(myUnitId)){
	    			for(Entry<Integer, Set<Integer>> attacking : beingAttackedBy.entrySet()){
	    				if(attacking.getValue().contains(myUnitId) && !deadEnemies.contains(attacking.getKey())){
	    					actions.put(myUnitId, Action.createCompoundAttack(myUnitId, attacking.getKey()));
	    					cumulativeReward.put(myUnitId, cumulativeReward.get(myUnitId) - 0.1);
	    				}else if(deadEnemies.contains(attacking.getKey())){
	    					hadEvent.add(myUnitId);
	    					inactiveUnits.add(myUnitId);
	    				}
	    			}
    			}else{
    				hadEvent.add(myUnitId);
    			}
    			/*if(myFootmen.contains(result.getKey())){
    				inactiveUnits.add(result.getKey());
    			}*/
    		}
    	}
    	
    	/*for(Entry<Integer, Integer> assignedMoves : lastAssignedAMove.entrySet()){
    		if(assignedMoves.getValue() >= 12){
    			hadEvent.add(assignedMoves.getKey());
    			inactiveUnits.add(assignedMoves.getKey());
    		}else{
    			lastAssignedAMove.put(assignedMoves.getKey(), assignedMoves.getValue() + 1);
    		}
    	}*/
    	
    	for(Integer myUnit : myFootmen){
    		if(attackedBy.get(myUnit) == attackedByStatus.NOT_ATTACKED_BY_TARGET){
    			inactiveUnits.add(myUnit);
    			hadEvent.add(myUnit);
    		}
    	}
    	if(explorationEpisode){
	    	for(Integer unitHadEvent : hadEvent){
	    		updateWeights(stateView, historyView, unitHadEvent);
	    	}
	    }else{
	    	for(Integer unitHadEvent : hadEvent){
	    		rewardsFromCurrentEpisode.add(cumulativeReward.get(unitHadEvent));
	    	}
	    }
    	inactiveUnits.removeAll(myDead);
    	for(Integer deadEnemy : deadEnemies){
    		beingAttackedBy.remove(deadEnemy);
    	}
    	
    	return inactiveUnits;
    }
    
    public double dotProduct(Double[] a, Double[] b){
		if(a.length != b.length){
			throw new IllegalArgumentException("The dimensions have to be equal!");
		}
		double sum = 0;
		for(int i = 0; i < a.length; i++){
			sum += a[i] * b[i];
		}
		return sum;
	}
    /**
     * Calculate the Q-Value for a given state action pair. The state in this scenario is the current
     * state view and the history of this episode. The action is the attacker and the enemy pair for the
     * SEPIA attack action.
     *
     * This returns the Q-value according to your feature approximation. This is where you will calculate
     * your features and multiply them by your current weights to get the approximate Q-value.
     *
     * @param stateView Current SEPIA state
     * @param historyView Episode history up to this point in the game
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman that your footman would be attacking
     * @return The approximate Q-value
     */
    public double calcQValue(State.StateView stateView,
                             History.HistoryView historyView,
                             int attackerId,
                             int defenderId) {
    	Double feature[] = calculateFeatureVector(stateView, historyView, attackerId, defenderId);
    	
        return dotProduct(weights,feature);
    }

    /**
     * Given a state and action calculate your features here. Please include a comment explaining what features
     * you chose and why you chose them.
     *
     * All of your feature functions should evaluate to a double. Collect all of these into an array. You will
     * take a dot product of this array with the weights array to get a Q-value for a given state action.
     *
     * It is a good idea to make the first value in your array a constant. This just helps remove any offset
     * from 0 in the Q-function. The other features are up to you. Many are suggested in the assignment
     * description.
     *
     * @param stateView Current state of the SEPIA game
     * @param historyView History of the game up until this turn
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman. The one you are considering attacking.
     * @return The array of feature function outputs.
     */
    public Double[] calculateFeatureVector(State.StateView stateView,
                                           History.HistoryView historyView,
                                           int attackerId,
                                           int defenderId) {
        Double[] fv = new Double[NUM_FEATURES];
        fv[0] = 1.0;
        
        UnitView at = stateView.getUnit(attackerId);
        UnitView df = stateView.getUnit(defenderId);
        double atHP = at == null? 0 : at.getHP();
        double dfHP = df == null? 0 : df.getHP();
        if(dfHP == 0 || atHP == 0){
        	fv[1] = 0.0;
        	fv[2] = 0.0;
        	fv[0] = 0.0;
        	//fv[3] = 0.0;
        }else{
        	int chebDist = Math.max(Math.abs(df.getXPosition() - at.getXPosition()),Math.abs(df.getYPosition() - at.getYPosition()));
        	fv[1] = getDistanceIndex(at.getXPosition(), at.getYPosition(), chebDist, stateView);// == 0 ? 10.0 : 0.0;
        	fv[2] = mostAttacked(defenderId);//beingAttackedBy.get(defenderId) != null && beingAttackedBy.get(defenderId).isEmpty() ? -1.0 : 1.0;
        	//fv[3] = atHP/at.getTemplateView().getBaseHealth() - dfHP/df.getTemplateView().getBaseHealth();
        }
       // System.out.println(fv[1]);
        
        //fv[2] = justAttacked(stateView, historyView,attackerId, defenderId);
        /*if(beingAttackedBy.get(defenderId) != null){
        	fv[3] = 1.0*beingAttackedBy.get(defenderId).size();
        }else{
        	fv[3] = 0.0;
        }
        
        /*fv[1] = (atHP + 1)/(atHP + dfHP + 1);
        
        if(dfHP == 0 || atHP == 0){
        	fv[2] = 0.0;
        	fv[5] = 0.0;
        }else{
        	int chebDist = Math.max(df.getXPosition() - at.getXPosition(),df.getYPosition() - at.getYPosition());
        	fv[2] = (double) chebDist;
        	fv[5] = getDistanceIndex(at.getXPosition(), at.getYPosition(), chebDist, stateView);
        }
        
        if(beingAttackedBy.get(defenderId) != null){
        	fv[2] = 1.0*beingAttackedBy.get(defenderId).size();
        }else{
        	fv[2] = 0.0;
        }
        
        if(dfHP!=0){
        	fv[4] = atHP/dfHP;
        }else{
        	fv[4] = 0.0;
        }
        */
        //fv[2] = justAttacked(stateView, historyView,attackerId, defenderId);
        
    	return fv;
    }

    private Double mostAttacked(int defenderId) {
		if(beingAttackedBy.get(defenderId) == null){
			return -1.0;
		}else{
			int enemyMostAttacked = -1;
			int numAttacking = Integer.MIN_VALUE;
			for(Entry<Integer, Set<Integer>> e : beingAttackedBy.entrySet()){
				if(e.getValue().size() > numAttacking){
					numAttacking = e.getValue().size();
					enemyMostAttacked = e.getKey();
				}
			}
			return enemyMostAttacked == defenderId? 1.0: -1.0;
		}
	}

	private Double justAttacked(StateView stateView, HistoryView historyView,
			int attackerId, int defenderId) {
    	for(int i = 1; i <= 3 && stateView.getTurnNumber() - i > 0; i++){
	    	for(DamageLog damageLog : historyView.getDamageLogs(stateView.getTurnNumber() - i)) {
	    		if(damageLog.getDefenderController() == playernum){
	    			int myUnit = damageLog.getDefenderID();
	    			int enemyId = damageLog.getAttackerID();
	    			if(myUnit == attackerId && enemyId == defenderId){
	    				return 1.0;
	    			}
	    		}
	    	}
    	}
    	return -1.0;
	}

	private Double getDistanceIndex(int x, int y, int chebDist, StateView stateView) {
    	TreeSet<Integer> sortedByDistance = new TreeSet<Integer>();
    	for(Integer enemyId : enemyFootmen){
    		UnitView en = stateView.getUnit(enemyId);
    		sortedByDistance.add(Math.max(Math.abs(en.getXPosition()) - x,Math.abs(en.getYPosition() - y)));
    	}
    	
    	int index = 0;
    	for(Integer dist : sortedByDistance){
    		if(dist == chebDist){
    			break;
    		}
    		index++;
    	}
    	//Collections.sort(sortedByDistance);
    	return index == 0? 1.0 : -1.0;//Math.ceil(((double) sortedByDistance.size())/2) - index;
	}

	/**
     * DO NOT CHANGE THIS!
     *
     * Prints the learning rate data described in the assignment. Do not modify this method.
     *
     * @param averageRewards List of cumulative average rewards from test episodes.
     */
    public void printTestData (List<Double> averageRewards) {
        System.out.println("");
        System.out.println("Games Played      Average Cumulative Reward");
        System.out.println("-------------     -------------------------");
        for (int i = 0; i < averageRewards.size(); i++) {
            String gamesPlayed = Integer.toString(10*i);
            String averageReward = String.format("%.2f", averageRewards.get(i));

            int numSpaces = "-------------     ".length() - gamesPlayed.length();
            StringBuffer spaceBuffer = new StringBuffer(numSpaces);
            for (int j = 0; j < numSpaces; j++) {
                spaceBuffer.append(" ");
            }
            System.out.println(gamesPlayed + spaceBuffer.toString() + averageReward);
        }
        System.out.println("");
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * This function will take your set of weights and save them to a file. Overwriting whatever file is
     * currently there. You will use this when training your agents. You will include th output of this function
     * from your trained agent with your submission.
     *
     * Look in the agent_weights folder for the output.
     *
     * @param weights Array of weights
     */
    public void saveWeights(Double[] weights) {
        File path = new File("agent_weights/weights.txt");
        // create the directories if they do not already exist
        path.getAbsoluteFile().getParentFile().mkdirs();

        try {
            // open a new file writer. Set append to false
            BufferedWriter writer = new BufferedWriter(new FileWriter(path, false));

            for (double weight : weights) {
                writer.write(String.format("%f\n", weight));
            }
            writer.flush();
            writer.close();
        } catch(IOException ex) {
            System.err.println("Failed to write weights to file. Reason: " + ex.getMessage());
        }
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * This function will load the weights stored at agent_weights/weights.txt. The contents of this file
     * can be created using the saveWeights function. You will use this function if the load weights argument
     * of the agent is set to 1.
     *
     * @return The array of weights
     */
    public Double[] loadWeights() {
        File path = new File("agent_weights/weights.txt");
        if (!path.exists()) {
            System.err.println("Failed to load weights. File does not exist");
            return null;
        }

        try {
            BufferedReader reader = new BufferedReader(new FileReader(path));
            String line;
            List<Double> weights = new LinkedList<>();
            while((line = reader.readLine()) != null) {
                weights.add(Double.parseDouble(line));
            }
            reader.close();

            return weights.toArray(new Double[weights.size()]);
        } catch(IOException ex) {
            System.err.println("Failed to load weights from file. Reason: " + ex.getMessage());
        }
        return null;
    }

    @Override
    public void savePlayerData(OutputStream outputStream) {

    }

    @Override
    public void loadPlayerData(InputStream inputStream) {

    }
}
