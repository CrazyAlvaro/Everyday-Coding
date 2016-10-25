// ag-grid-enterprise v4.2.10
var __extends = (this && this.__extends) || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
};
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var main_1 = require("ag-grid/main");
var abstractColumnDropPanel_1 = require("./abstractColumnDropPanel");
var svgFactory = main_1.SvgFactory.getInstance();
var PivotColumnsPanel = (function (_super) {
    __extends(PivotColumnsPanel, _super);
    function PivotColumnsPanel(horizontal) {
        _super.call(this, horizontal);
    }
    PivotColumnsPanel.prototype.passBeansUp = function () {
        _super.prototype.setBeans.call(this, {
            eventService: this.eventService,
            context: this.context,
            loggerFactory: this.loggerFactory,
            dragAndDropService: this.dragAndDropService
        });
        var localeTextFunc = this.gridOptionsWrapper.getLocaleTextFunc();
        var emptyMessage = localeTextFunc('pivotColumnsEmptyMessage', 'Drag here to pivot');
        var title = localeTextFunc('pivots', 'Pivots');
        _super.prototype.init.call(this, {
            dragAndDropIcon: main_1.DragAndDropService.ICON_GROUP,
            iconFactory: svgFactory.createPivotIcon,
            emptyMessage: emptyMessage,
            title: title
        });
        this.addDestroyableEventListener(this.eventService, main_1.Events.EVENT_COLUMN_PIVOT_CHANGED, this.refreshGui.bind(this));
    };
    PivotColumnsPanel.prototype.isColumnDroppable = function (column) {
        var columnPivotable = !column.getColDef().suppressPivot;
        var columnNotAlreadyPivoted = !this.columnController.isColumnPivoted(column);
        return columnPivotable && columnNotAlreadyPivoted;
    };
    PivotColumnsPanel.prototype.removeColumns = function (columns) {
        var _this = this;
        var columnsPivoted = main_1.Utils.filter(columns, function (column) { return _this.columnController.isColumnPivoted(column); });
        this.columnController.removePivotColumns(columnsPivoted);
    };
    PivotColumnsPanel.prototype.addColumns = function (columns) {
        this.columnController.addPivotColumns(columns);
    };
    PivotColumnsPanel.prototype.getExistingColumns = function () {
        return this.columnController.getPivotColumns();
    };
    __decorate([
        main_1.Autowired('columnController'), 
        __metadata('design:type', main_1.ColumnController)
    ], PivotColumnsPanel.prototype, "columnController", void 0);
    __decorate([
        main_1.Autowired('eventService'), 
        __metadata('design:type', main_1.EventService)
    ], PivotColumnsPanel.prototype, "eventService", void 0);
    __decorate([
        main_1.Autowired('gridOptionsWrapper'), 
        __metadata('design:type', main_1.GridOptionsWrapper)
    ], PivotColumnsPanel.prototype, "gridOptionsWrapper", void 0);
    __decorate([
        main_1.Autowired('context'), 
        __metadata('design:type', main_1.Context)
    ], PivotColumnsPanel.prototype, "context", void 0);
    __decorate([
        main_1.Autowired('loggerFactory'), 
        __metadata('design:type', main_1.LoggerFactory)
    ], PivotColumnsPanel.prototype, "loggerFactory", void 0);
    __decorate([
        main_1.Autowired('dragAndDropService'), 
        __metadata('design:type', main_1.DragAndDropService)
    ], PivotColumnsPanel.prototype, "dragAndDropService", void 0);
    __decorate([
        main_1.PostConstruct, 
        __metadata('design:type', Function), 
        __metadata('design:paramtypes', []), 
        __metadata('design:returntype', void 0)
    ], PivotColumnsPanel.prototype, "passBeansUp", null);
    PivotColumnsPanel = __decorate([
        main_1.Bean('pivotColumnsPanel'), 
        __metadata('design:paramtypes', [Boolean])
    ], PivotColumnsPanel);
    return PivotColumnsPanel;
})(abstractColumnDropPanel_1.AbstractColumnDropPanel);
exports.PivotColumnsPanel = PivotColumnsPanel;
